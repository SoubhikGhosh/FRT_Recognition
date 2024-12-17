import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import faiss
import mediapipe as mp
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MediaPipe BlazeFace
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_face(image, target_size=(160, 160)):
    """
    Extract faces using MediaPipe BlazeFace.
    
    Parameters:
    - image: The image to process (OpenCV format).
    - target_size: The target size to resize detected faces.
    
    Returns:
    - boxes: The bounding boxes of detected faces.
    - cropped_faces: A list of cropped face images.
    """
    # Convert the image to RGB as MediaPipe expects RGB input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize the face detection model
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        # Perform face detection
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None, None
        
        boxes = []
        cropped_faces = []

        # Extract faces from detected bounding boxes
        for detection in results.detections:
            # Get the bounding box for each detected face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            x2 = int((bboxC.xmin + bboxC.width) * iw)
            y2 = int((bboxC.ymin + bboxC.height) * ih)
            boxes.append([x1, y1, x2, y2])

            # Crop the face from the image
            face = image[y1:y2, x1:x2]
            if face.size == 0:  # Ensure face is valid
                continue
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
    """
    Builds a face database by extracting face embeddings from all images 
    in subdirectories named after individuals, with a progress bar.
    """
    database = {}
    
    # Walk through the image folder
    for root, dirs, files in os.walk(image_folder):
        # Skip if no images in the directory
        if not files:
            continue
        
        # Use the last folder name as the person's name
        person_name = os.path.basename(root)
        embeddings_list = []
        
        # Use tqdm to show progress on files in the current directory
        for file in tqdm(files, desc=f"Processing {person_name}", ncols=100, unit="file"):
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
            
            # Load the image
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                continue

            # Extract faces and encode
            boxes, faces = extract_face(image)
            if faces:
                embeddings = encode_faces(faces, facenet)
                embeddings_list.extend(embeddings)
            else:
                print(f"Face not detected in {image_path}")

        # Store all embeddings for the person in the database
        if embeddings_list:
            database[person_name] = embeddings_list
            print(f"Added {len(embeddings_list)} embeddings for {person_name}")
        else:
            print(f"No valid faces found for {person_name}")
    print(f"loaded faces: {len(database)}")
    return database

def recognize_faces(image, database, threshold=0.7):
    """
    Detect and recognize faces in an image by comparing embeddings 
    with all known embeddings in the database.
    """
    # Step 1: Detect faces in the image
    boxes, faces = extract_face(image)
    if not faces:
        return image, []

    # Step 2: Encode detected faces into embeddings
    embeddings = encode_faces(faces, facenet)

    results = []

    # Step 3: Match query embeddings against database embeddings
    for embedding in embeddings:
        best_match_name = None
        best_match_score = -1  # Initialize with a low score
        
        for person_name, stored_embeddings in database.items():
            stored_embeddings = np.vstack(stored_embeddings)  # Stack stored embeddings
            
            # Normalize embeddings
            stored_embeddings = stored_embeddings / np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
            query_embedding = embedding / np.linalg.norm(embedding)

            # Compute cosine similarity with all stored embeddings
            similarities = np.dot(stored_embeddings, query_embedding)
            max_similarity = np.max(similarities)

            if max_similarity > best_match_score:
                best_match_name = person_name
                best_match_score = max_similarity
        
        # Compare against the threshold
        if best_match_score >= threshold:
            results.append((best_match_name, best_match_score))
        else:
            results.append((None, best_match_score))

    # Step 4: Annotate the image
    annotated_image = image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated_image, results

def recognize_faces_faiss(input_image, database, threshold=0.6, nlist=4):
    """
    Optimized face recognition using FAISS with approximate nearest neighbor search.
    """
    # Step 1: Extract and crop faces from the input image
    boxes, faces = extract_face(input_image)
    if not faces:
        return input_image, []

    # Step 2: Generate embeddings for the detected faces
    embeddings = encode_faces(faces, facenet)

    # Step 3: Prepare FAISS index
    # Flatten all embeddings from the database into a single list
    all_db_embeddings = []
    embedding_to_name_map = []  # Keeps track of which person each embedding belongs to

    for name, embedding_list in database.items():
        for embedding in embedding_list:
            all_db_embeddings.append(embedding)
            embedding_to_name_map.append(name)

    # Convert to numpy array and normalize for cosine similarity
    all_db_embeddings = np.array(all_db_embeddings).astype('float32')
    faiss.normalize_L2(all_db_embeddings)

    # Create FAISS index for approximate search
    dimension = all_db_embeddings.shape[1]
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, nlist)
    index.train(all_db_embeddings)
    index.add(all_db_embeddings)

    # Step 4: Normalize input embeddings and search using FAISS
    faiss.normalize_L2(embeddings)
    k = 1  # Top-1 match for each detected face
    distances, indices = index.search(embeddings, k)

    # Step 5: Process the FAISS results
    results = []
    for dist, idx in zip(distances[:, 0], indices[:, 0]):
        if dist > threshold and idx < len(embedding_to_name_map):
            results.append((embedding_to_name_map[idx], dist))  # Map embedding back to person's name
        else:
            results.append((None, dist))

    # Step 6: Annotate the image with the recognition results
    annotated_image = input_image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated_image, results
