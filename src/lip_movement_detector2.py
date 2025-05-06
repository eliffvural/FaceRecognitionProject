import dlib
import cv2
import numpy as np
import os

class LipMovementDetector2:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor dosyası bulunamadı: {predictor_path}")
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, prev_frame, curr_frame, face_coords, frame):
        x, y, w, h = face_coords

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(curr_gray, rect)

        lip_points = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]
        for (px, py) in lip_points:
            cv2.circle(frame, (px, py), 1, (255, 0, 0), -1)

        lip_x = min(p[0] for p in lip_points)
        lip_y = min(p[1] for p in lip_points)
        lip_w = max(p[0] for p in lip_points) - lip_x
        lip_h = max(p[1] for p in lip_points) - lip_y

        if lip_w == 0 or lip_h == 0:
            return False

        prev_lip = prev_gray[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]
        curr_lip = curr_gray[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]

        if prev_lip.shape != curr_lip.shape or prev_lip.size == 0:
            return False

        # 1. Dudak açıklığı (üst iç - alt iç nokta)
        try:
            upper_inner = shape.part(62).y
            lower_inner = shape.part(66).y
            lip_opening = abs(lower_inner - upper_inner)
        except:
            lip_opening = 0

        # 2. Optik akış (hareket büyüklüğü)
        flow = cv2.calcOpticalFlowFarneback(prev_lip, curr_lip, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_movement = np.mean(magnitude)

        # 3. Piksel farkı
        diff = cv2.absdiff(prev_lip, curr_lip)
        sum_diff = np.sum(diff)
        area = lip_w * lip_h
        normalized_movement = sum_diff / area

        # Eşikler (deneysel olarak test edilmeli)
        talking = False
        if lip_opening > 5 and flow_movement > 0.8:
            talking = True

        # Debug çıktısı
        print(f"Lip opening: {lip_opening}, Flow: {flow_movement:.2f}, NormDiff: {normalized_movement:.2f}")

        return talking
