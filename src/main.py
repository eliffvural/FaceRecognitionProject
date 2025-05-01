import cv2
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from lip_movement_detector import LipMovementDetector

def main():
    try:
        # Modülleri başlat
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        lip_detector = LipMovementDetector()

        # Kamerayı aç
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera açılamadı! Başka bir uygulamanın kamerayı kullanıp kullanmadığını kontrol edin.")
            return

        # Kamera çözünürlüğünü kontrol et ve ayarla
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Pencereyi oluştur ve boyutlandır
        cv2.namedWindow("Yuz Tanima", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Yuz Tanima", 640, 480)

        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Çerçeve alınamadı, kamera bağlantısını kontrol edin!")
                break

            # Yüzleri tespit et
            faces = detector.detect_faces(frame)

            for (x, y, x2, y2) in faces:
                face = frame[y:y2, x:x2]
                if face.size == 0:
                    continue

                # Yüzü tanı (face_recognition için koordinatlar: y, x2, y2, x)
                name = recognizer.recognize_face(face, frame, (y, x, x2, y2))
                if name is None:
                    continue

                # Çerçeve çiz
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Konuşma tespiti
                if prev_frame is not None:
                    if lip_detector.detect(prev_frame[y:y2, x:x2], frame[y:y2, x:x2], (0, 0, x2 - x, y2 - y)):
                        print(f"{name} konuşuyor")
                        cv2.putText(frame, "Konusuyor", (x, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            prev_frame = frame.copy()

            # Frame'i göster
            cv2.imshow("Yuz Tanima", frame)
            cv2.moveWindow("Yuz Tanima", 0, 0)

            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Hata oluştu: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()