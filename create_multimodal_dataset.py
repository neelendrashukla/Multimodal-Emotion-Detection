import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Custom Multimodal Dataset class
class MultimodalDataset(Dataset):
    def __init__(self, text_data, audio_data, image_data, labels):
        self.text_data = text_data
        self.audio_data = audio_data
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text_data[idx],
            'audio': self.audio_data[idx],
            'image': self.image_data[idx],
            'label': self.labels[idx]
        }

def get_text_embeddings(texts, batch_size=32, max_length=128):
    """Convert text strings to BERT embeddings."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, 
                          truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings, dim=0)

def inspect_and_align_dataset(file_path, modality_name, convert_text=False, reference_labels=None):
    """Inspect and align dataset, optionally matching to reference labels."""
    data = torch.load(file_path)
    print(f"\nInspecting {modality_name} dataset ({file_path}):")
    print(f"  Type: {type(data)}")

    if not isinstance(data, tuple) or len(data) < 2:
        raise TypeError(f"{modality_name} dataset is not a tuple with at least 2 items, got {type(data)} with {len(data)} items!")

    features, labels = data[0], data[1]
    print(f"  Features: type={type(features)}, length={len(features)}")
    print(f"  Labels: type={type(labels)}, shape={labels.shape if hasattr(labels, 'shape') else len(labels)}")

    # Handle text features (list of strings)
    if modality_name == "Text" and convert_text and isinstance(features, list):
        if not all(isinstance(t, str) for t in features):
            raise ValueError(f"{modality_name} features contain non-string elements!")
        print(f"  Converting {modality_name} text features to BERT embeddings...")
        features = get_text_embeddings(features)
        print(f"  Converted features: type={type(features)}, shape={features.shape}")

    # Verify features and labels are tensors
    if not torch.is_tensor(features) or not torch.is_tensor(labels):
        raise TypeError(f"{modality_name} dataset: Features or labels are not tensors after processing!")

    # Verify sample count
    if len(features) != 15000:
        raise ValueError(f"{modality_name} dataset has {len(features)} samples, expected 15000!")
    if len(labels) != 15000:
        raise ValueError(f"{modality_name} dataset labels has {len(labels)} samples, expected 15000!")

    # Check class distribution
    unique, counts = torch.unique(labels, return_counts=True)
    print(f"  Classes: {unique.tolist()}")
    print(f"  Class counts: {counts.tolist()}")
    if len(unique) != 6:
        raise ValueError(f"{modality_name} dataset has {len(unique)} classes, expected 6!")
    if not torch.all(counts == 2500):
        raise ValueError(f"{modality_name} dataset class counts are uneven: {counts.tolist()}")

    # Check for NaN or Inf values
    if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
        raise ValueError(f"{modality_name} dataset contains NaN or Inf values!")

    # Align with reference labels (e.g., audio labels)
    if reference_labels is not None:
        print(f"  Aligning {modality_name} samples to reference labels...")
        indices = []
        ref_counts = torch.bincount(reference_labels, minlength=6)
        curr_counts = torch.bincount(labels, minlength=6)
        if not torch.all(ref_counts == curr_counts):
            raise ValueError(f"{modality_name} class distribution does not match reference!")
        
        for cls in range(6):
            ref_cls_indices = (reference_labels == cls).nonzero(as_tuple=True)[0]
            curr_cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
            if len(ref_cls_indices) != len(curr_cls_indices):
                raise ValueError(f"Class {cls} has mismatched sample counts!")
            indices.append(curr_cls_indices[:len(ref_cls_indices)])
        indices = torch.cat(indices)
        features = features[indices]
        labels = labels[indices]
        print(f"  Aligned {modality_name} samples to reference labels.")

    return features, labels

def align_and_create_multimodal_dataset(text_path, audio_path, image_path, output_path):
    # Inspect and align datasets (use audio as reference since audio and image are aligned)
    audio_data, audio_labels = inspect_and_align_dataset(audio_path, "Audio")
    image_data, image_labels = inspect_and_align_dataset(image_path, "Image")
    text_data, text_labels = inspect_and_align_dataset(text_path, "Text", convert_text=True, reference_labels=audio_labels)

    # Verify label alignment
    if not (torch.all(text_labels == audio_labels) and torch.all(audio_labels == image_labels)):
        raise ValueError("Labels are not aligned after processing! Please provide metadata for proper alignment.")

    # Data quality checks
    for data, modality in [(text_data, 'Text'), (audio_data, 'Audio'), (image_data, 'Image')]:
        if data.dtype not in [torch.float32, torch.float64]:
            print(f"Warning: {modality} data has dtype {data.dtype}, converting to float32.")
            data = data.to(dtype=torch.float32)

    # Create the multimodal dataset
    multimodal_dataset = MultimodalDataset(text_data, audio_data, image_data, audio_labels)

    # Save the aligned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'text': text_data,
        'audio': audio_data,
        'image': image_data,
        'labels': audio_labels
    }, output_path)

    print(f"\nMultimodal dataset created and saved to {output_path}")
    print(f"Total samples: {len(multimodal_dataset)}")
    print(f"Data quality checks passed. Ready for training.")

    return multimodal_dataset

if __name__ == "__main__":
    # File paths
    text_path = r"D:\mood_detection\data\processed_multimodal_balanced\text.pt"
    audio_path = r"D:\mood_detection\data\processed_multimodal_balanced\audio.pt"
    image_path = r"D:\mood_detection\data\processed_multimodal_balanced\images.pt"
    output_path = r"D:\mood_detection\data\processed_multimodal_balanced\multimodal_dataset.pt"

    # Create and align the multimodal dataset
    try:
        dataset = align_and_create_multimodal_dataset(text_path, audio_path, image_path, output_path)

        # Create a DataLoader for training
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Test the dataset
        print("\nTesting DataLoader (first batch):")
        for batch in dataloader:
            print("  Batch keys:", batch.keys())
            print("  Text shape:", batch['text'].shape)
            print("  Audio shape:", batch['audio'].shape)
            print("  Image shape:", batch['image'].shape)
            print("  Label shape:", batch['label'].shape)
            print("  Sample labels:", batch['label'][:5].tolist())
            break

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check the dataset files and their structure or provide metadata for alignment.")