from flask import Flask, request, render_template, jsonify, Response
import cv2
import os
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.lip_movement_detector import LipMovementDetector
from src.lip_movement_detector2 import LipMovementDetector2
import urllib.request
import numpy as np
import yt_dlp
import time
import json
from deepface import DeepFace
from collections import defaultdict
import time

app = Flask(__name__)

# Yüklenen dosyalar için geçici klasör
UPLOAD_FOLDER = 'data/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_video_stream(video_source):
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    recognizer.reset_session()  # Yeni video için oturumu sıfırla
    try:
        lip_detector = LipMovementDetector()
    except Exception as e:
        yield json.dumps({"error": f"Dudak hareketi dedektörü başlatılamadı: {str(e)}"}) + "\n"
        return

    # Video kaynağını al (URL veya dosya yolu)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        yield json.dumps({"error": "Video açılamadı!"}) + "\n"
        return

    # FPS değerini al (konuşma süresini hesaplamak için)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Varsayılan FPS (tahmini)
    print(f"Videonun FPS değeri: {fps}")

    # Toplam çerçeve sayısını ve video süresini al
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames <= 0:
        total_frames = float('inf')  # Eğer bilinmiyorsa, sonsuz olarak kabul et
    total_duration = total_frames / fps if total_frames != float('inf') else 0
    print(f"Videonun toplam süresi: {total_duration:.2f} saniye (Toplam çerçeve: {total_frames})")

    # Çıktı videosu için
    output_dir = os.path.join("data", "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(output_dir, "processed_video.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Tüm videoyu işle
    max_frames = float('inf')

    prev_frame = None
    frame_count = 0
    current_faces = []
    speaking_times = {}  # Her kişinin konuşma süresini sakla (saniye cinsinden)
    speaking_frames = {}  # Her kişinin konuşma çerçeve sayısını sakla
    last_speaking = {}  # Son konuşma anını takip et

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame_count >= max_frames:
                break

            # Her 3 çerçeveye bir yüz algılama ve tanıma yap
            if frame_count % 3 == 0:
                faces = detector.detect_faces(frame)
                current_faces = []

                for (x, y, x2, y2) in faces:
                    face = frame[y:y2, x:x2]
                    if face.size == 0:
                        continue

                    name = recognizer.recognize_face(face, frame, (y, x, x2, y2))
                    if name is None:
                        continue  # Yüz tanınamazsa atla

                    current_faces.append((name, (x, y, x2, y2)))

                    # Yeni bir kişi eklenirse konuşma süresini başlat
                    if name not in speaking_times:
                        speaking_times[name] = 0
                        speaking_frames[name] = 0
                        last_speaking[name] = 0

                # Algılanan yüzleri logla
                print(f"Çerçeve {frame_count}: {len(current_faces)} yüz algılandı. Kişiler: {[name for name, _ in current_faces]}")

            # Tespit edilen yüzleri işle
            for name, (x, y, x2, y2) in current_faces:
                face = frame[y:y2, x:x2]
                if face.size == 0:
                    continue

                # Konuşma tespiti
                if prev_frame is not None:
                    speaking = lip_detector.detect(prev_frame[y:y2, x:x2], frame[y:y2, x:x2], (x, y, x2 - x, y2 - y), frame)
                    if speaking:
                        speaking_frames[name] += 1
                        last_speaking[name] = frame_count
                        print(f"Çerçeve {frame_count}: {name} konuşuyor, hareket algılandı.")
                    else:
                        # 15 çerçeve (yaklaşık 0.5 saniye) içinde konuşma devam ediyorsa, konuşma süresini sıfırlama
                        if frame_count - last_speaking.get(name, 0) <= 15:
                            speaking_frames[name] += 1
                        else:
                            speaking_frames[name] = 0

                    # Konuşma süresini güncelle (her çerçeve için)
                    if speaking_frames[name] > 0:
                        speaking_times[name] += 1 / fps  # Saniye cinsinden
                        print(f"Çerçeve {frame_count}: {name} için konuşma süresi güncellendi: {speaking_times[name]:.2f} saniye")

                # Görselleştirme: Kişi isimlerini ve konuşma durumunu çerçeve üzerine yaz
                cv2.putText(frame, name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if speaking_frames[name] > 0:
                    cv2.putText(frame, f"{name} konuşuyor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            prev_frame = frame.copy()
            frame_count += 1
            out.write(frame)  # Çerçeveyi çıktı videosuna yaz

            # Her 20 çerçevede bir güncel sonuçları gönder
            if frame_count % 20 == 0:
                progress_percent = (frame_count / total_frames) * 100 if total_frames != float('inf') else 0
                yield json.dumps({
                    "progress": f"Çerçeve {frame_count} işlendi ({progress_percent:.1f}%)",
                    "results": speaking_times
                }) + "\n"

    except Exception as e:
        yield json.dumps({"error": f"Video işlenirken hata oluştu: {str(e)}"}) + "\n"
    finally:
        cap.release()
        out.release()  # Çıktı videosunu kapat
        cv2.destroyAllWindows()

    # Nihai sonuçlardan önce çok kısa konuşma sürelerini filtrele (0.5 saniyeden kısa)
    filtered_speaking_times = {name: time for name, time in speaking_times.items() if time >= 0.5}

    # Nihai sonuçları gönder
    print(f"Filtrelenmiş nihai konuşma süreleri: {filtered_speaking_times}")
    total_speaking_time = sum(filtered_speaking_times.values())
    print(f"Toplam konuşma süresi: {total_speaking_time:.2f} saniye (Videonun toplam süresi: {total_duration:.2f} saniye)")
    yield json.dumps({
        "progress": "İşleme tamamlandı!",
        "results": filtered_speaking_times
    }) + "\n"

from collections import defaultdict
import time



def generate_frames():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    lip_detector = LipMovementDetector2()

    global unknown_faces, unknown_counter
    global talking_durations, talking_start_times, currently_talking

    talking_durations = defaultdict(float)     # Toplam konuşma süresi
    talking_start_times = {}                   # Aktif konuşma başlangıçları
    face_appearance_times = {}  # face_id -> first_seen_time

    currently_talking = set() 

    unknown_faces = {}
    unknown_counter = 0

    prev_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = detector.detect_faces(frame)

        detected_names = set()  # O an karede görünen kişiler

        for (x, y, x2, y2) in faces:
            face = frame[y:y2, x:x2]
            name = recognizer.recognize_face(face, frame, (y, x, x2, y2))

            if name is None:
                face_id = hash(face.tobytes())

                if face_id not in face_appearance_times:
                    face_appearance_times[face_id] = time.time()

                elapsed = time.time() - face_appearance_times[face_id]
                if elapsed >= 2.0 and face_id not in unknown_faces:
                    unknown_counter += 1
                    unknown_faces[face_id] = f"Person{chr(64 + unknown_counter)}"

                name = unknown_faces.get(face_id, None)
                if name is None:
                    continue  # Hala 2 saniye dolmadıysa işlemi atla


            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if prev_frame is not None:
                is_talking = lip_detector.detect(prev_frame, frame, (x, y, x2 - x, y2 - y), frame)

                if is_talking:
                    cv2.putText(frame, "Konusuyor", (x, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    if name not in currently_talking:
                        talking_start_times[name] = time.time()
                        currently_talking.add(name)

                else:
                    if name in currently_talking:
                        start = talking_start_times.pop(name, None)
                        if start is not None:
                            duration = time.time() - start
                            if duration >= 1.0:  # En az 1 saniye konuşmuş olmalı
                                talking_durations[name] += duration
                        currently_talking.remove(name)

        



            try:
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                cv2.putText(frame, f"Duygu: {dominant_emotion}", (x, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                print(f"{name} Duygu Durumu: {dominant_emotion}")
            except Exception as e:
                print(f"Duygu analizi hatası: {e}")

        # Konuşan kişi artık görünmüyorsa, konuşmayı kesmiş say
        for name in list(currently_talking):
            if name not in detected_names:
                start = talking_start_times.pop(name, None)
                if start is not None:
                    talking_durations[name] += time.time() - start
                currently_talking.remove(name)

        prev_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Ana sayfa
@app.route('/')
def home():
    return render_template('home.html')

# Kayıtlı video sayfası
@app.route('/recorded')
def recorded():
    return render_template('index.html')

# Canlı video sayfası
@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/camera_stream', methods=['GET', 'OPTIONS'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

from flask import jsonify

@app.route('/speech_data')
def speech_data():
    # Güncel konuşan varsa, süresini ekle
    current_time = time.time()
    updated_durations = {}

    for name, start_time in talking_start_times.items():
        updated_durations[name] = talking_durations.get(name, 0) + (current_time - start_time)

    # Diğerlerini de ekle
    for name, total_time in talking_durations.items():
        if name not in updated_durations:
            updated_durations[name] = total_time

    # JSON olarak gönder
    response = [{"kisi": name, "sure": round(seconds, 1)} for name, seconds in updated_durations.items()]
    return jsonify(response)


@app.route('/process', methods=['POST'])
def process():
    video_source = None
    temp_file_path = None

    if 'video_file' in request.files:
        file = request.files['video_file']
        if file.filename == '':
            return jsonify({"error": "Dosya seçilmedi!"})
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_file_path)
        video_source = temp_file_path

    elif 'video_url' in request.form:
        url = request.form['video_url']
        if not url:
            return jsonify({"error": "URL girilmedi!"})

        # YouTube URL’si mi kontrol et
        if "youtube.com" in url or "youtu.be" in url:
            try:
                ydl_opts = {
                    'format': 'best',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_source = info['url']  # Doğrudan video akış URL’sini al
            except Exception as e:
                print(f"İndirme hatası: {str(e)}")
                return jsonify({"error": f"YouTube videosu indirilemedi: {str(e)}"})
        else:
            video_source = url

    else:
        return jsonify({"error": "Geçersiz istek!"})

    def generate(video_source=video_source, temp_file_path=temp_file_path):
        try:
            for result in process_video_stream(video_source):
                yield result
        except Exception as e:
            yield json.dumps({"error": f"Stream sırasında hata oluştu: {str(e)}"}) + "\n"
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return Response(generate(video_source=video_source, temp_file_path=temp_file_path), mimetype='application/json')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
