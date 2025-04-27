import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeatureExtractor().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.known_faces = {}  # {person_name: [embedding_list]}
        self.next_person_id = 0  # Yeni kişiler için ID
        self.threshold = 0.7  # Tanıma için güven eşiği

    def get_embedding(self, face):
        # Yüzü işle
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)

        # Embedding çıkar
        with torch.no_grad():
            embedding = self.model(face_tensor).cpu().numpy()
        return embedding

    def recognize_face(self, face):
        embedding = self.get_embedding(face)

        # Bilinen yüzlerle karşılaştır
        min_dist = float('inf')
        recognized_person = "Bilinmeyen Kişi"
        for person_name, embeddings in self.known_faces.items():
            for known_embedding in embeddings:
                dist = np.linalg.norm(embedding - known_embedding)
                if dist < min_dist and dist < self.threshold:
                    min_dist = dist
                    recognized_person = person_name

        # Eğer bilinmeyen bir yüzse, kaydet
        if recognized_person == "Bilinmeyen Kişi":
            new_person_name = f"person_{chr(65 + self.next_person_id)}"  # Örneğin, person_A
            self.save_face(face, new_person_name, embedding)
            recognized_person = new_person_name
            self.next_person_id += 1

        return recognized_person

    def save_face(self, face, person_name, embedding):
        # Yüzü kaydet
        save_dir = f"../data/known_faces/{person_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Yüzü kaydet (örneğin, face_1.jpg)
        face_count = len(os.listdir(save_dir)) + 1
        save_path = os.path.join(save_dir, f"face_{face_count}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        print(f"Yüz kaydedildi: {save_path}")

        # Embedding’i kaydet
        if person_name not in self.known_faces:
            self.known_faces[person_name] = []
        self.known_faces[person_name].append(embedding)