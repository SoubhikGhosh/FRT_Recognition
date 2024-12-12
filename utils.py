import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
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
        if ext.lower() not in ['.jpg', '.png', '.jpeg']:
            continue
        image = cv2.imread(os.path.join(image_folder, file))
        faces = extract_face(image, mtcnn)[1]
        if faces:
            embeddings = encode_faces(faces, facenet)
            database[name] = embeddings[0]
            print(name)

        else:
            print (f"face not detected for {name}" )
    return database

def recognize_faces(image, database, threshold=0.7):
    
    # Step 1: Detect faces in the image
    boxes, faces = extract_face(image, mtcnn)
    if not faces:
        return image, []

    # Step 2: Encode detected faces into embeddings
    embeddings = encode_faces(faces, facenet)
    
    # Step 3: Precompute database embeddings and normalize
    db_embeddings = np.array(list(database.values())).astype('float32')
    db_names = list(database.keys())
    
    # Normalize embeddings if required
    if db_embeddings.shape[0] > 0:
        db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Step 4: Match using NumPy (faster for small datasets)
    results = []
    for embedding in embeddings:
        # Compute cosine similarity
        similarities = np.dot(db_embeddings, embedding)
        best_match_idx = np.argmax(similarities)
        best_match_score = similarities[best_match_idx]
        
        if best_match_score > threshold:
            results.append((db_names[best_match_idx], best_match_score))
        else:
            results.append((None, best_match_score))
    
    # Step 5: Annotate the image
    annotated_image = image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated_image, results