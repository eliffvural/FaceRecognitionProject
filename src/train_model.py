import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from face_detector import FaceDetector

# Veri seti sınıfı
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.face_detector = FaceDetector()
        self.images = []
        self.labels = []
        self.label_map = {}  # Kişi isimlerini indekslere eşleştir

        # Veri setini yükle
        for idx, person_name in enumerate(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            self.label_map[idx] = person_name
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                self.images.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        # Görseli yükle
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Yüzü tespit et ve kırp
        faces = self.face_detector.detect_faces(image)
        if len(faces) == 0:
            # Yüz bulunamazsa, varsayılan bir boyut kullan
            image = cv2.resize(image, (224, 224))
        else:
            x, y, x2, y2 = faces[0]
            image = image[y:y2, x:x2]
            image = cv2.resize(image, (224, 224))

        # Dönüşümleri uygula
        if self.transform:
            image = self.transform(image)

        return image, label

# Basit bir CNN modeli
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # Yüz vektörü (embedding) çıkarma için özellik katmanını kullan
    def get_embedding(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        return x

def train_model(data_dir, num_epochs=10):
    # Dönüşümler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Veri setini yükle
    dataset = FaceDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Modeli oluştur
    num_classes = len(dataset.label_map)
    model = FaceRecognitionModel(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Kayıp fonksiyonu ve optimizasyon
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Modeli eğit
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

    # Modeli kaydet
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_map': dataset.label_map,
    }, "../models/face_recognition_model.pth")
    print("Model kaydedildi: ../models/face_recognition_model.pth")

if __name__ == "__main__":
    data_dir = "../data/dataset/"  # Veri setinin bulunduğu klasör
    train_model(data_dir, num_epochs=10)