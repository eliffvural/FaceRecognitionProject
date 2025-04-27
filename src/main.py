import cv2
import os
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from lip_movement_detector import LipMovementDetector

class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.lip_detector = LipMovementDetector()
        self.cap = cv2.VideoCapture(0)  # Varsayılan kamera (0)

        if not self.cap.isOpened():
            raise ValueError("Kamera açılamadı. Lütfen kameranızın bağlı olduğundan emin olun.")

        # Çıktı videosu için
        output_dir = "../data/output/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "output_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(self.cap.get(3)), int(self.cap.get(4))))

    def process_camera(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı.")
                break

            # Yüzleri tespit et
            faces = self.face_detector.detect_faces(frame)
            for (x, y, x2, y2) in faces:
                face = frame[y:y2, x:x2]
                if face.size == 0:
                    continue

                # Yüzü tanı
                person_name = self.face_recognizer.recognize_face(face)

                # Konuşma tespiti
                is_speaking, lip_points = self.lip_detector.detect_lip_movement(frame)
                status = f"{person_name} konuşuyor" if is_speaking else f"{person_name}"

                # Yüzü çerçevele ve isim yaz
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Dudak noktalarını çiz (isteğe bağlı)
                if lip_points:
                    for point in lip_points:
                        cv2.circle(frame, point, 2, (0, 0, 255), -1)

            # Frame’i göster ve kaydet
            cv2.imshow("Face Recognition", frame)
            self.out.write(frame)

            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"Çıktı videosu kaydedildi: {output_path}")

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.process_camera()