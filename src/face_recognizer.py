import os
import cv2
import numpy as np
import face_recognition

class FaceRecognizer:
    def __init__(self):
        self.known_faces = {}  # {person_name: [encoding_list]}
        self.next_person_id = 0  # Yeni kişiler için ID
        self.tolerance = 0.5  # face_recognition için eşleşme toleransı (mesafe eşiği)
        self.unknown_count = {}  # Tutarlılık kontrolü için sayaç
        self.min_frames_for_new_person = 10  # Yeni kişi kaydetmeden önce 10 çerçeve tutarlılık

    def get_encoding(self, face, frame, face_coords):
        # face_coords: (y, x2, y2, x) formatında olmalı (face_recognition için)
        y, x, x2, y2 = face_coords
        encoding = face_recognition.face_encodings(frame, [(y, x2, y2, x)])
        if len(encoding) == 0:
            return None
        return encoding[0]

    def recognize_face(self, face, frame, face_coords):
        encoding = self.get_encoding(face, frame, face_coords)
        if encoding is None:
            return None

        # Veri tabanı boşsa ilk kişiyi doğrudan kaydet
        if not self.known_faces:
            person_name = "personA"  # İlk kişi her zaman "personA" olacak
            print(f"Yeni kişi kaydediliyor: {person_name}")
            self.save_face(face, person_name, encoding)
            self.next_person_id = 1  # İlk kişi kaydedildi, sonraki kişi için ID artır
            self.unknown_count = {}  # Sayacı sıfırla
            return person_name

        # Bilinen yüzlerle karşılaştır
        all_encodings = []
        name_mapping = []
        for person_name, encodings in self.known_faces.items():
            for enc in encodings:
                all_encodings.append(enc)
                name_mapping.append(person_name)

        if not all_encodings:
            return None

        # face_recognition ile eşleşme kontrolü yap
        matches = face_recognition.compare_faces(all_encodings, encoding, tolerance=self.tolerance)
        face_distances = face_recognition.face_distance(all_encodings, encoding)

        recognized_person = None
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                recognized_person = name_mapping[best_match_index]
                print(f"{recognized_person} ile mesafe: {face_distances[best_match_index]}")

        # Tanıma başarılıysa unknown_count'u sıfırla
        if recognized_person:
            if recognized_person in self.unknown_count:
                self.unknown_count[recognized_person] = 0
            return recognized_person

        # Eşleşme bulunamadı, tutarlılık kontrolü yap
        temp_person_name = f"temp_{self.next_person_id}"
        if temp_person_name not in self.unknown_count:
            self.unknown_count[temp_person_name] = 0
        self.unknown_count[temp_person_name] += 1

        # Eğer belirli sayıda çerçevede tutarlı bir şekilde eşleşme bulunamadıysa yeni kişi kaydet
        if self.unknown_count[temp_person_name] >= self.min_frames_for_new_person:
            new_person_name = f"person{chr(65 + self.next_person_id)}"  # Örneğin, personB, personC...
            print(f"Yeni kişi kaydediliyor: {new_person_name}")
            self.save_face(face, new_person_name, encoding)
            self.next_person_id += 1
            self.unknown_count.pop(temp_person_name)  # Sayacı sıfırla
            return new_person_name

        print(f"Eşleşme bulunamadı, tutarlılık kontrolü: {self.unknown_count[temp_person_name]}/{self.min_frames_for_new_person}")
        return None  # Henüz yeni kişi kaydetmiyoruz

    def save_face(self, face, person_name, encoding):
        # Yüzü kaydet
        save_dir = f"data/known_faces/{person_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Yüzü kaydet (en fazla 3 görüntü sakla)
        face_count = len(os.listdir(save_dir)) + 1
        if face_count > 3:  # Eğer 3’ten fazla görüntü varsa, eski bir görüntüyü sil
            old_face = os.path.join(save_dir, f"face_{face_count - 3}.jpg")
            if os.path.exists(old_face):
                os.remove(old_face)
            face_count -= 1

        save_path = os.path.join(save_dir, f"face_{face_count}.jpg")
        success = cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        if success:
            print(f"Yüz kaydedildi: {save_path}")
        else:
            print(f"Yüz kaydedilemedi: {save_path}")

        # Encoding’i kaydet
        if person_name not in self.known_faces:
            self.known_faces[person_name] = []
        self.known_faces[person_name].append(encoding)