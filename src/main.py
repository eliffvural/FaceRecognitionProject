import cv2
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from lip_movement_detector import LipMovementDetector
from deepface import DeepFace

def main():
    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        lip_detector = LipMovementDetector()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera açılamadı!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("Yuz Tanima", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Yuz Tanima", 640, 480)

        prev_frame = None
        frame_count = 0
        analyze_interval = 5  # her 5 karede bir analiz

        emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 255, 0),
            'fear': (255, 0, 255),
            'happy': (0, 255, 255),
            'sad': (255, 0, 0),
            'surprise': (255, 255, 0),
            'neutral': (128, 128, 128)
        }

        emotions = {}  # yüz ID'si yerine x-y konumuna göre emotion saklama

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Kare alınamadı!")
                break

            frame_count += 1
            faces = detector.detect_faces(frame)

            if frame_count % analyze_interval == 0:
                try:
                    analysis_results = DeepFace.analyze(
                        img_path=frame,
                        actions=['emotion'],
                        detector_backend='opencv',
                        enforce_detection=False
                    )

                    # Her yüz için dominant duyguyu sakla
                    analysis_faces = analysis_results if isinstance(analysis_results, list) else [analysis_results]
                    emotions = {}
                    for result in analysis_faces:
                        region = result['region']
                        emotion = result['dominant_emotion'].lower()
                        emotions[(region['x'], region['y'], region['w'], region['h'])] = emotion

                except Exception as e:
                    print(f"DeepFace hatası: {e}")

            for (x, y, x2, y2) in faces:
                face = frame[y:y2, x:x2]
                if face.size == 0:
                    continue

                name = recognizer.recognize_face(face, frame, (y, x, x2, y2))
                if name is None:
                    continue

                # Konuşma tespiti
                if prev_frame is not None:
                    if lip_detector.detect(prev_frame[y:y2, x:x2], frame[y:y2, x:x2], (0, 0, x2 - x, y2 - y)):
                        cv2.putText(frame, "Konusuyor", (x, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Duygu çerçevesi: daire
                matched_emotion = None
                for (ex, ey, ew, eh), emo in emotions.items():
                    if abs(ex - x) < 30 and abs(ey - y) < 30:  # yakın konum eşleştirmesi
                        matched_emotion = emo
                        break

                if matched_emotion:
                    color = emotion_colors.get(matched_emotion, (255, 255, 255))
                    center = (x + (x2 - x) // 2, y + (y2 - y) // 2)
                    radius = int(max(x2 - x, y2 - y) * 0.6)
                    cv2.circle(frame, center, radius, color, 2)
                    cv2.putText(frame, matched_emotion.capitalize(), (x, y - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    # normal çerçeve
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # İsmi yaz
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            prev_frame = frame.copy()

            cv2.imshow("Yuz Tanima", frame)
            cv2.moveWindow("Yuz Tanima", 0, 0)

            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Hata oluştu: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()