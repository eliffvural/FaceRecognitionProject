import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, prototxt_path=None, model_path=None):
        # Varsayılan yollar
        if prototxt_path is None:
            prototxt_path = os.path.join(os.path.dirname(__file__), "../models/deploy.prototxt")
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "../models/res10_300x300_ssd_iter_140000.caffemodel")

        # Dosyaların varlığını kontrol et
        if not os.path.exists(prototxt_path):
            raise FileNotFoundError(f"Prototxt dosyası bulunamadı: {prototxt_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect_faces(self, frame, confidence_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                faces.append((x, y, x2, y2))
        return faces