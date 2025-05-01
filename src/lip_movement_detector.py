import cv2
import numpy as np
import dlib
import os

class LipMovementDetector:
    def __init__(self):
        # Dlib’in yüz landmark tahmincisini yükle
        predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Landmark tahmin dosyası bulunamadı: {predictor_path}. Lütfen indirin: https://github.com/davisking/dlib-models")
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_lip_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return False, None

        # İlk yüzü al
        face = faces[0]
        landmarks = self.predictor(gray, face)

        # Dudak noktalarını al (48-67 indeksleri dudaklar için)
        lip_points = []
        for i in range(48, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            lip_points.append((x, y))

        # Dudak noktalarının doğru boyutta olduğundan emin ol
        if len(lip_points) < 20:  # 48-67 arası toplam 20 nokta olmalı
            print("Dudak noktaları eksik, konuşma tespiti atlanıyor.")
            return False, lip_points

        # Dudak açıklığını hesapla (örneğin, iç dudaklar arasındaki mesafe)
        try:
            lip_height = abs(lip_points[14][1] - lip_points[18][1])  # 62-48=14, 66-48=18
            lip_width = abs(lip_points[12][0] - lip_points[16][0])   # 60-48=12, 64-48=16
            lip_ratio = lip_height / lip_width if lip_width > 0 else 0

            # Eşik değeri (konuşma için dudak hareketi)
            threshold = 0.3
            is_speaking = lip_ratio > threshold
        except IndexError as e:
            print(f"Dudak noktaları hesaplanırken hata: {e}")
            return False, lip_points

        return is_speaking, lip_points