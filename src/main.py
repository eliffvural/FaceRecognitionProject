import cv2
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from lip_movement_detector import LipMovementDetector

def main():
    # Modülleri başlat
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    lip_detector = LipMovementDetector()

    # Videoyu oku
    cap = cv2.VideoCapture("../data/input/input_video.mp4")
    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return

    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Yüzleri tespit et
        faces = detector.detect_faces(frame)

        for (x, y, x2, y2) in faces:
            face = frame[y:y2, x:x2]
            if face.size == 0:
                continue

            # Yüzü tanı
            name = recognizer.recognize_face(face)
            if name is None:
                continue

            # Çerçeve çiz
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Konuşma tespiti
            if prev_frame is not None:
                if lip_detector.detect(prev_frame, frame, (x, y, x2 - x, y2 - y)):
                    print(f"{name} konuşuyor")
                    cv2.putText(frame, "Konusuyor", (x, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        prev_frame = frame.copy()

        # Frame'i göster
        cv2.imshow("Yuz Tanima", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()