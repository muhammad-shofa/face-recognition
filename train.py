import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from db import connect_db

# Inisialisasi InsightFace
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0)  # ganti ke -1 kalau tanpa GPU

# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0, det_size=(320, 240))  # kecilkan dari default (640,640)

# Ambil nama user dari input
user_name = input("Masukkan nama user: ")

# Jalankan kamera
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Arahkan wajah ke kamera dan tekan 's' untuk menyimpan.")

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(frame_rgb)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)

    cv2.imshow("Training - Tekan 's' untuk simpan", frame)
    key = cv2.waitKey(1)
    if key == ord('s') and faces:
        encoding = faces[0].embedding
        encoding_blob = pickle.dumps(encoding)

        db = connect_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO users (name, face_encoding) VALUES (%s, %s)", (user_name, encoding_blob))
        db.commit()
        db.close()
        print(f"Wajah {user_name} berhasil disimpan ke database.")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
