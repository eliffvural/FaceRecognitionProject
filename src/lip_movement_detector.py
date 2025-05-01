import cv2
import numpy as np

class LipMovementDetector:
    def __init__(self):
        self.movement_threshold = 3000  # Eşiği artırdık, daha az hassas
        self.min_frames_for_speaking = 3  # Konuşma için 3 çerçeve üst üste hareket şart
        self.speaking_counter = {}  # Her yüz ID’si için konuşma sayacı

    def detect(self, prev_frame, curr_frame, face_coords):
        # Yüz ID’sini oluştur (konum bilgisiyle benzersiz bir kimlik oluşturuyoruz)
        face_id = f"{face_coords[0]}_{face_coords[1]}"  # x ve y koordinatları ile ID

        # Yüz bölgesinin alt yarısını al (dudak bölgesi tahmini)
        h = face_coords[3]  # Yüz yüksekliği
        lip_region_prev = prev_frame[int(h/2):h, :]
        lip_region_curr = curr_frame[int(h/2):h, :]

        # Gri tonlamaya çevir
        lip_region_prev = cv2.cvtColor(lip_region_prev, cv2.COLOR_BGR2GRAY)
        lip_region_curr = cv2.cvtColor(lip_region_curr, cv2.COLOR_BGR2GRAY)

        # Gürültüyü azaltmak için bulanıklaştırma uygula
        lip_region_prev = cv2.GaussianBlur(lip_region_prev, (5, 5), 0)
        lip_region_curr = cv2.GaussianBlur(lip_region_curr, (5, 5), 0)

        # Farkı hesapla
        diff = cv2.absdiff(lip_region_prev, lip_region_curr)
        diff_sum = np.sum(diff)

        # Hareket eşiğini kontrol et
        if diff_sum > self.movement_threshold:
            # Eğer bu yüz için sayaç yoksa başlat
            if face_id not in self.speaking_counter:
                self.speaking_counter[face_id] = 0
            self.speaking_counter[face_id] += 1
        else:
            # Hareket yoksa sayacı sıfırla
            self.speaking_counter[face_id] = 0

        # Konuşma için minimum çerçeve sayısını kontrol et
        if self.speaking_counter.get(face_id, 0) >= self.min_frames_for_speaking:
            return True
        return False