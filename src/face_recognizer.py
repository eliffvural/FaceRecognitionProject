import numpy as np
from deepface import DeepFace


#yüz tanıma ve isimlendirme
class FaceRecognizer:
    def __init__(self):
        self.known_faces = {}  # Yüz vektörleri ve isimler
        self.face_id_counter = 0

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

        return recognized_name