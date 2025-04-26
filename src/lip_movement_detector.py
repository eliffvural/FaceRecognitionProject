import cv2
import numpy as np


#konusma tespiti icin dudak hareketlerini algılayan sınıf
class LipMovementDetector:
    def detect(self, prev_frame, current_frame, face_coords, threshold=30):
        x, y, w, h = face_coords
        lip_region_y = y + int(h * 0.6)
        lip_region_h = int(h * 0.3)
        lip_region = current_frame[lip_region_y:lip_region_y + lip_region_h, x:x + w]
        prev_lip_region = prev_frame[lip_region_y:lip_region_y + lip_region_h, x:x + w]

        if lip_region.size == 0 or prev_lip_region.size == 0:
            return False

        lip_region_gray = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
        prev_lip_region_gray = cv2.cvtColor(prev_lip_region, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(lip_region_gray, prev_lip_region_gray)
        return np.mean(diff) > threshold