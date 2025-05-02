import dlib
import cv2
import numpy as np
import os

class LipMovementDetector:
    def __init__(self):
        # Dosyanın tam yolunu belirle
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor dosyası bulunamadı: {predictor_path}")
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, prev_frame, curr_frame, face_coords, frame):
        x, y, w, h = face_coords

        # Grayscale’e çevir
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Dudak noktalarını bul
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(curr_gray, rect)
        lip_points = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]

        # Dudak noktalarını çerçeve üzerinde işaretle (hata ayıklama için)
        for (px, py) in lip_points:
            cv2.circle(frame, (px, py), 1, (255, 0, 0), -1)

        # Dudak bölgesini sınırlandır (dudak noktalarına göre)
        lip_x = min(p[0] for p in lip_points) - x
        lip_y = min(p[1] for p in lip_points) - y
        lip_w = max(p[0] for p in lip_points) - x - lip_x
        lip_h = max(p[1] for p in lip_points) - y - lip_y

        # Dudak bölgesini kırp
        prev_lip_region = prev_gray[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]
        curr_lip_region = curr_gray[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]

        if prev_lip_region.shape != curr_lip_region.shape or prev_lip_region.size == 0 or curr_lip_region.size == 0:
            return False

        # Farkı hesapla
        diff = cv2.absdiff(prev_lip_region, curr_lip_region)
        movement = np.sum(diff)

        # Hareketi normalizasyon yaparak eşikle karşılaştır
        area = lip_w * lip_h
        if area == 0:
            return False
        normalized_movement = movement / area

        # Dinamik eşik değeri: Yanlış pozitifleri azaltmak için eşik artırıldı
        threshold = max(8, min(20, area * 0.03))  # Daha yüksek eşik
        print(f"Dudak hareketi: normalized_movement={normalized_movement:.2f}, threshold={threshold:.2f}, area={area}")
        return normalized_movement > threshold