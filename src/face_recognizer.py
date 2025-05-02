import cv2
import face_recognition
import os
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.session_face_encodings = []  # Oturum boyunca bilinmeyen yüzleri sakla
        self.session_face_names = []  # Oturum boyunca bilinmeyen yüzlerin isimlerini sakla
        self.person_counter = 0  # Yeni kişiler için sayaç (A, B, ... olarak isimlendirmek için)
        self.load_known_faces()

    def load_known_faces(self):
        known_faces_dir = "data/known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)

    def recognize_face(self, face_image, frame, face_coords):
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        if len(encodings) == 0:
            return None

        face_encoding = encodings[0]

        # Önce bilinen yüzlerle eşleştir
        if self.known_face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                return name

        # Bilinmeyen bir yüzse, oturumdaki yüzlerle eşleştir
        if self.session_face_encodings:
            matches = face_recognition.compare_faces(self.session_face_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.session_face_names[first_match_index]
                return name

        # Yeni bir kişi olarak kaydet (personA, personB, ... şeklinde)
        self.person_counter += 1
        name = f"person{chr(64 + self.person_counter)}"  # 1 -> personA, 2 -> personB, ...
        self.session_face_encodings.append(face_encoding)
        self.session_face_names.append(name)
        print(f"Yeni kişi eklendi: {name}")
        return name

    def reset_session(self):
        # Yeni bir video için oturumu sıfırla
        self.session_face_encodings = []
        self.session_face_names = []
        self.person_counter = 0