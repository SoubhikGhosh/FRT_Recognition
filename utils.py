import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face(image, mtcnn, target_size=(160, 160)):
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None, None
    cropped_faces = []
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        face = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face, target_size)
        cropped_faces.append(face_resized)
    return boxes, cropped_faces

def encode_faces(faces, facenet):
    embeddings = []
    for face in faces:
        face = torch.tensor(face.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        embedding = facenet(face).detach().cpu().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

def build_face_database(image_folder):
    database = {}
    for file in os.listdir(image_folder):
        name, ext = os.path.splitext(file)
        if ext.lower() not in ['.jpg', '.png']:
            continue
        image = cv2.imread(os.path.join(image_folder, file))
        faces = extract_face(image, mtcnn)[1]
        if faces:
            embeddings = encode_faces(faces, facenet)
            database[name] = embeddings[0]
    return database

def recognize_faces(image, database, threshold=0.6):
    boxes, faces = extract_face(image, mtcnn)
    if not faces:
        return image, []

    embeddings = encode_faces(faces, facenet)
    db_embeddings = np.array(list(database.values())).astype('float32')
    db_names = list(database.keys())

    index = faiss.IndexFlatIP(db_embeddings.shape[1])
    faiss.normalize_L2(db_embeddings)
    index.add(db_embeddings)

    faiss.normalize_L2(embeddings)
    distances, indices = index.search(embeddings, 1)

    # results = []
    # for box, (dist, idx) in zip(boxes, zip(distances[:, 0], indices[:, 0])):
    #     if dist > threshold and idx < len(db_names):
    #         results.append((box, db_names[idx], dist))
    #     else:
    #         results.append((box, "Unknown", dist))

    results = []
    for dist, idx in zip(distances[:, 0], indices[:, 0]):
        if dist > threshold and idx < len(db_names):
            results.append((db_names[idx], dist))
        else:
            results.append((None, dist))

    annotated_image = image.copy()

    # for (box, name, score) in results:
    #     x1, y1, x2, y2 = [int(b) for b in box]
    #     label = f"{name} ({score:.2f})"
    #     color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    #     cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
    #     cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for (box, (name, score)) in zip(mtcnn.detect(image)[0], results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
    return annotated_image, results
