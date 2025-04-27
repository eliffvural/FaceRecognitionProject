import numpy as np
import json
import os
from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, embeddings_file="../data/known_faces.json"):
        self.embeddings_file = embeddings_file
        self.known_faces = {}
        self.face_id_counter = 0

        # Önceki vektörleri dosyadan yükle (varsa)
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r') as f:
                loaded_data = json.load(f)
                self.known_faces = {tuple(map(float, k.split(','))): v for k, v in loaded_data.items()}
                # Son kişi ID'sini bul
                if self.known_faces:
                    self.face_id_counter = max(ord(name[-1]) - 64 for name in self.known_faces.values())

    def save_embeddings(self):
        # Vektörleri dosyaya kaydet (anahtarları string'e çevir)
        embeddings_to_save = {','.join(map(str, k)): v for k, v in self.known_faces.items()}
        with open(self.embeddings_file, 'w') as f:
            json.dump(embeddings_to_save, f)

    def recognize_face(self, face):
        try:
            embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        except Exception:
            return None

        # Bu yüzü daha önce gördük mü?
        recognized_name = None
        for known_embedding, name in self.known_faces.items():
            distance = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
            if distance < 10:  # Eşik değeri
                recognized_name = name
                break

        # Yeni yüzse kaydet
        if recognized_name is None:
            self.face_id_counter += 1
            recognized_name = f"Kişi {chr(64 + self.face_id_counter)}"  # A, B, C...
            self.known_faces[tuple(embedding)] = recognized_name
            print(f"Yeni yüz kaydedildi: {recognized_name}")
            self.save_embeddings()  # Yeni yüzü dosyaya kaydet

        return recognized_name