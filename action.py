import cv2
import insightface
import numpy as np
import pickle
from db import connect_db 

# Inisialisasi model insightface
# model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# model = insightface.app.FaceAnalysis(name="buffalo_l")
# model.prepare(ctx_id=0, det_size=(320, 240))  # ⬅️ Lebih ringan


# Ambil data dari database
db = connect_db()
cursor = db.cursor()
cursor.execute("SELECT name, face_encoding FROM users")  # Sesuaikan nama kolom
results = cursor.fetchall()
db.close()

# Siapkan list nama dan embeddings
known_names = []
known_embeddings = []

for name, embedding_blob in results:
    embedding = pickle.loads(embedding_blob)
    known_names.append(name)
    known_embeddings.append(embedding)

known_embeddings = np.array(known_embeddings)

# Mulai kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 5 != 0:
        continue  # ⬅️ Lewati setiap frame ganjil (bisa ubah ke %3 kalau terlalu cepat)

    if not ret:
        break

    faces = model.get(frame)
    for face in faces:
        embedding = face.embedding

        # Cosine similarity
        similarities = np.dot(known_embeddings, embedding) / (
            np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-5
        )

        max_index = np.argmax(similarities)
        similarity = similarities[max_index]

        if similarity > 0.5:
            name = known_names[max_index]
        else:
            name = "Unrecognized"

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - InsightFace", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # cv2.imshow("Face Recognition - InsightFace", frame)
    # if cv2.waitKey(10) & 0xFF == ord('q'):  # ⬅️ 10 ms delay untuk

cap.release()
cv2.destroyAllWindows()
