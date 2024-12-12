import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss
import os
# import nmslib


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

# def recognize_faces_nmslib(input_image, database, mtcnn, facenet, threshold):
#     """
#     Recognizes faces in the input image by comparing with the database using NMSLIB.
    
#     Parameters:
#     - input_image: The image to process.
#     - database: A dictionary of precomputed embeddings (name -> embedding).
#     - mtcnn: The face detector.
#     - facenet: The face recognition model.
#     - threshold: Similarity threshold for recognition.
    
#     Returns:
#     - Annotated image with recognition results.
#     """
#     # Extract and crop faces
#     faces = extract_face(input_image, mtcnn)
#     if not faces:
#         return input_image, []

#     # Generate embeddings for detected faces
#     embeddings = encode_faces(faces, facenet)
    
#     # Prepare NMSLIB index
#     # Convert database to a format suitable for NMSLIB
#     db_embeddings = np.array(list(database.values())).astype('float32')
#     db_names = list(database.keys())

#     # Create NMSLIB index
#     index = nmslib.init(method='hnsw', space='cosinesimil')
#     for idx, embedding in enumerate(db_embeddings):
#         index.addDataPoint(idx, embedding)
#     index.createIndex({'post': 2}, print_progress=False)

#     results = []
#     for embedding in embeddings:
#         # Query the nearest neighbor
#         nearest_neighbors = index.knnQuery(embedding, k=1)
#         idx, dist = nearest_neighbors[0][0], 1 - nearest_neighbors[1][0]  # Convert to similarity score
#         if dist > threshold and idx < len(db_names):
#             results.append((db_names[idx], dist))
#         else:
#             results.append((None, dist))
    
#     # Annotate image
#     annotated_image = input_image.copy()
#     for (box, (name, score)) in zip(mtcnn.detect(input_image)[0], results):
#         x1, y1, x2, y2 = [int(b) for b in box]
#         label = f"{name} ({score:.2f})" if name else "Unknown"
#         color = (0, 255, 0) if name else (0, 0, 255)
#         cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
    
#     return annotated_image, results

