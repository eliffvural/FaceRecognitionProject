import cv2
import dlib
import os

class FaceDetector:
    def __init__(self):
        # Dlib’in yüz dedektörünü kullan
        self.detector = dlib.get_frontal_face_detector()
        print("Yüz dedektörü başlatıldı.")

    def detect_faces(self, frame):
        # Grayscale’e çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Yüzleri algıla
        faces = self.detector(gray)
        face_coords = []
        for face in faces:
            x = face.left()
            y = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # Koordinatların çerçeve sınırları içinde olduğundan emin ol
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            face_coords.append((x, y, x2, y2))
        return face_coords