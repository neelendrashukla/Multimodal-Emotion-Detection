import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import cv2
from imutils.video import VideoStream
import pyaudio
import librosa
from transformers import BertTokenizer, BertModel
from collections import defaultdict
import csv
import time

# Multimodal Model (same as training code)
class MultimodalModel(nn.Module):
    def __init__(self, num_classes=6):
        super(MultimodalModel, self).__init__()
        # Text branch (BERT embeddings: [768])
        self.text_branch = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64)
        )
        
        # Audio branch (Spectrograms: [94, 40])
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.audio_lstm = nn.LSTM(256 * 11 * 5, 512, batch_first=True, num_layers=2, dropout=0.4)
        self.audio_fc = nn.Linear(512, 128)
        
        # Image branch (ResNet-18: [3, 224, 224])
        self.image_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_branch.fc = nn.Linear(self.image_branch.fc.in_features, 128)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, text, audio, image):
        # Text forward
        text_out = self.text_branch(text)
        
        # Audio forward
        audio = audio.unsqueeze(1)
        audio = self.audio_conv(audio)
        audio = audio.view(audio.size(0), -1, audio.size(1) * audio.size(2) * audio.size(3))
        audio, _ = self.audio_lstm(audio)
        audio_out = self.audio_fc(audio[:, -1, :])
        
        # Image forward
        image_out = self.image_branch(image)
        
        # Fusion
        combined = torch.cat([text_out, audio_out, image_out], dim=1)
        out = self.fusion(combined)
        return out

# Advanced Recommendation System
class AdvancedRecommendation:
    def __init__(self):
        self.mood_items = {
            "angry": ["breathing_exercise", "slow_music", "calm_podcast"],
            "disgust": ["motivational_quote", "upbeat_music", "inspirational_video"],
            "fear": ["calming_music", "guided_meditation", "relaxing_sound"],
            "happy": ["party_music", "dance_track", "upbeat_playlist"],
            "sad": ["soft_music", "emotional_song", "inspirational_book"],
            "neutral": ["chill_playlist", "ambient_music", "neutral_podcast"]
        }
        self.trending_items = ["latest_hit_song", "viral_video", "new_podcast"]
        self.history = defaultdict(list)

    def get_collaborative_suggestion(self, mood):
        if os.path.exists(r'F:\mood_detection\user_history.csv'):
            with open(r'F:\mood_detection\user_history.csv', 'r') as f:
                reader = csv.DictReader(f)
                similar_items = defaultdict(int)
                for row in reader:
                    if row['mood'] == mood:
                        similar_items[row['item']] += int(row['rating'])
                return max(similar_items, key=similar_items.get) if similar_items else None
        return None

    def get_recommendation(self, mood, history_weight=0.6, trend_weight=0.2, collab_weight=0.2):
        collab_item = self.get_collaborative_suggestion(mood)
        content_items = self.mood_items.get(mood, ["default_suggestion"])
        trend_item = np.random.choice(self.trending_items)

        recommendations = []
        for item in content_items:
            score = history_weight * (item in self.history[mood]) + trend_weight * (item == trend_item) + collab_weight * (item == collab_item)
            recommendations.append((item, score))
        if collab_item:
            recommendations.append((collab_item, collab_weight))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendation = recommendations[0][0] if recommendations else content_items[0]
        self.history[mood].append(top_recommendation)
        return top_recommendation

# Real-time testing function
def real_time_test(model, device):
    model.eval()
    cap = VideoStream(src=0).start()
    time.sleep(2.0)
    mood_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "neutral"}
    recommender = AdvancedRecommendation()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)

    def get_audio_input():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        data = b''
        for _ in range(4):
            data += stream.read(1024, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)
        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)), 'constant')
        mfcc = np.transpose(librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512))
        if mfcc.shape[0] < 94:
            pad_width = ((0, 94 - mfcc.shape[0]), (0, 0))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        stream.stop_stream()
        stream.close()
        p.terminate()
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def get_text_input(text):
        inputs = tokenizer(text.lower(), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze()

    def get_image(frame):
        transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform(frame).unsqueeze(0).to(device)

    while True:
        frame = cap.read()
        if frame is None:
            break
        image = get_image(frame)
        audio = get_audio_input().to(device)
        user_input = input("Your feeling: ").strip().lower()
        if user_input == "exit":
            break
        user_text = user_input if user_input.startswith("i am feeling") else f"i am feeling {user_input}" if user_input else "i am feeling neutral"
        text = get_text_input(user_text)

        with torch.no_grad():
            output = model(text.unsqueeze(0), audio, image)
            probabilities = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, pred].item()
            mood = mood_map[pred]
            recommendation = recommender.get_recommendation(mood)

        current_time = time.strftime("%H:%M:%S")
        print(f"\nTime: {current_time}")
        print(f"Webcam: Active")
        print(f"Audio: Captured")
        print(f"Text: {user_text}")
        print(f"Mood: {mood} (Confidence: {confidence:.2f})")
        print(f"Recommendation: {recommendation}")

    cap.stop()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalModel(num_classes=6).to(device)
    model.load_state_dict(torch.load(r'F:\mood_detection\model\best_model.pt'))
    real_time_test(model, device)