import cv2
import pickle
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import faiss
import mediapipe as mp
from tqdm import tqdm
import base64
from sklearn.preprocessing import normalize
import albumentations as A
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MediaPipe BlazeFace
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def get_next_filename(directory, name):
    """
    Generates the next filename in sequence for the given name.
    Example: If 'name_0001.jpg' exists, it returns 'name_0002.jpg'.
    """
    counter = 1
    while True:
        file_name = f"{name}_{counter:04d}.jpg"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def convert_to_float(value):
    """ Convert NumPy float32 to Python native float """
    if isinstance(value, np.float32):
        return float(value)
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(v) for v in value]
    return value

# Define the augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.3),
        A.RandomCrop(width=200, height=200, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ])

def save_and_process_image(name, base64_image, save_dir, database, facenet=None):
    """
    Save the image to a folder, process the face, update the in-memory and .pkl database.
    
    Parameters:
    - name: Name of the person (used for folder and labeling).
    - base64_image: Base64-encoded image string.
    - save_dir: Base directory to store images.
    - database: In-memory face database dictionary.
    """
    try:
        # Create user directory
        user_dir = os.path.join(save_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        # Generate the next file name
        file_path = get_next_filename(user_dir, name)

        # Decode and save the base64 image
        image_data = base64.b64decode(base64_image.split(",")[1])
        with open(file_path, "wb") as f:
            f.write(image_data)
        print(f"Image saved at {file_path}")

        # Read the saved image
        image = cv2.imread(file_path)
        if image is None:
            return {"error": "Invalid image data"}

        # Apply augmentations in real-time before processing the face
        augmentation_pipeline = get_augmentation_pipeline()
        augmented_image = augmentation_pipeline(image=image)["image"]

        # Detect and encode faces from the augmented image
        _, faces = extract_face(augmented_image)
        if not faces:
            return {"error": "No faces detected in the image"}

        embeddings = encode_faces(faces, facenet)

        # Update the database
        if name not in database:
            print("Name not in database, creating new record.")
            database[name] = [embeddings]  # Store as a list of embeddings
        else:
            print(f"Name {name} found in database, updating record.")
            
            # Recalculate the average embedding from all images in the folder
            print(f"Recalculating average embeddings for {name}")
            all_embeddings = []
            for filename in os.listdir(user_dir):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                    image_path = os.path.join(user_dir, filename)
                    image = cv2.imread(image_path)
                    
                    # Apply the same augmentation to this image
                    augmented_image = augmentation_pipeline(image=image)["image"]
                    
                    _, faces = extract_face(augmented_image)
                    if faces:
                        embeddings = encode_faces(faces, facenet)
                        all_embeddings.append(embeddings)

            # Recalculate the average of all embeddings
            if all_embeddings:
                averaged_embeddings = np.mean(np.array(all_embeddings), axis=0)
                database[name] = averaged_embeddings  # Update the database with the new average

        # Save the updated database to a .pkl file
        save_database(database)
        print("Updated .pkl file with the new entry.")

        return {"message": "Image processed and database updated successfully", "file_path": file_path}

    except Exception as e:
        print(f"Error during image processing: {e}")
        return {"error": str(e)}

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

        cv2.destroyAllWindows()  # If any windows were created during processing
        return boxes, cropped_faces

def encode_faces(faces, facenet):
    """
    Encodes the detected faces into embeddings using the FaceNet model.
    
    Parameters:
    - faces: List of detected face images (numpy arrays).
    - facenet: Pretrained FaceNet model.
    - device: The device (cpu or cuda) to run the model on.
    
    Returns:
    - embeddings: Numpy array of encoded face embeddings.
    """
    embeddings = []
    for face in faces:
        # Ensure the face is in the correct format (H, W, C -> C, H, W)
        face = torch.tensor(face.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        # Forward pass to get the embedding
        with torch.no_grad():  # Disable gradient calculation
            embedding = facenet(face)  # Get the embedding from the FaceNet model

        # Convert to numpy and flatten if necessary (e.g., (1, 128) -> (128,))
        embedding = embedding.detach().cpu().numpy().flatten()
        embeddings.append(embedding)

    # Stack all embeddings vertically and return
    embeddings = np.vstack(embeddings)
    
    # Check consistency of shapes for debugging
    if len(set([emb.shape[0] for emb in embeddings])) > 1:
        print(f"Warning: Inconsistent embedding shapes: {[emb.shape[0] for emb in embeddings]}")
        
    return embeddings


DATABASE_FILE = "face_database.pkl"  # File to store face database

def save_database(database, file_path=DATABASE_FILE):
    """ Save the face database to a file. """
    with open(file_path, 'wb') as f:
        pickle.dump(database, f)
    print(f"Face database saved to {file_path}")

def load_database(file_path=DATABASE_FILE):
    """ Load the face database from a file. """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            database = pickle.load(f)
        print(f"Face database loaded from {file_path}")
        return database
    else:
        print(f"No saved database found at {file_path}. Building a new one.")
        return {}

def build_face_database(image_folder):
    """
    Builds a face database by extracting face embeddings from all images 
    in subdirectories named after individuals, with a global progress bar.
    """
    database = {}

    # Check if we already have a saved database
    saved_database = load_database()
    if saved_database:
        return saved_database  # Return the existing database
    
    # Gather all image paths in a list for global progress tracking
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in ['.jpg', '.png', '.jpeg']:  # Only consider image files
                image_paths.append(os.path.join(root, file))
    
    # Create a tqdm progress bar for all images
    with tqdm(total=len(image_paths), desc="Loading database", ncols=100, unit="file") as pbar:
        # Process each image
        for image_path in image_paths:
            # Extract person's name from the directory (folder name)
            person_name = os.path.basename(os.path.dirname(image_path))
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                pbar.update(1)
                continue

            # Extract faces and encode
            boxes, faces = extract_face(image)
            if faces:
                embeddings = encode_faces(faces, facenet)

                # Store embeddings in the database for the person
                if person_name not in database:
                    database[person_name] = []
                database[person_name].extend(embeddings)
            else:
                print(f"Face not detected in {image_path}")
            
            # Update progress bar
            pbar.update(1)

    print(f"Loaded faces for {len(database)} individuals.")

    database = compute_average_embeddings(database)
    save_database(database)
    return database

def compute_average_embeddings(database):
    return {name: np.mean(np.array(embeddings), axis=0) for name, embeddings in database.items()}

def recognize_faces(image, database, threshold=0.7, confidence_margin=0.0001):
    """
    Detect and recognize faces in an image by comparing embeddings 
    with all known embeddings in the database.
    """

    print("Starting to recognize faces...")

    # Step 1: Detect faces in the image
    boxes, faces = extract_face(image)
    if not faces:
        print("No faces detected.")
        return image, []

    print(f"Detected {len(faces)} face(s). Encoding faces...")
    # Step 2: Encode detected faces into embeddings
    embeddings = encode_faces(faces, facenet)
    if len(embeddings) == 0:
        print("No embeddings generated.")
        return image, []

    # Normalize query embeddings
    embeddings = np.array([emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb for emb in embeddings])

    results = []

    print("Matching faces...")
    # Step 3: Match query embeddings against database embeddings
    for i, embedding in enumerate(embeddings):
        best_match_name = None
        best_match_score = -1  # Initialize with a low score
        second_best_score = -1  # Track second-best score for confidence margin

        for person_name, stored_embeddings in database.items():
            # Ensure stored_embeddings is a 2D array
            stored_embeddings = np.array(stored_embeddings)
            if stored_embeddings.ndim == 1:
                stored_embeddings = stored_embeddings[np.newaxis, :]  # Convert to 2D if needed
            
            # Normalize stored embeddings
            stored_embeddings = stored_embeddings / np.linalg.norm(stored_embeddings, axis=1, keepdims=True)

            # Flatten both stored embeddings and the query embedding to 1D
            embedding_flat = embedding.flatten()  # Flatten query embedding to (512,)
            stored_embeddings_flat = stored_embeddings.flatten()  # Flatten stored embeddings to (512,)

            # Compute cosine similarities between stored embeddings and the query embedding
            similarities = np.dot(stored_embeddings_flat, embedding_flat)  # Dot product between 1D vectors
            max_similarity = similarities

            if max_similarity > best_match_score:
                second_best_score = best_match_score  # Update second-best score
                best_match_name = person_name
                best_match_score = max_similarity
            elif max_similarity > second_best_score:
                second_best_score = max_similarity

        # Confidence margin check: Ensure best match is significantly better than second-best
        if best_match_score >= threshold and (best_match_score - second_best_score) >= confidence_margin:
            print(f"Face {i + 1}: Matched with {best_match_name} (Score: {best_match_score:.2f})")
            results.append((best_match_name, best_match_score))
        else:
            print(f"Face {i + 1}: No reliable match (Score: {best_match_score:.2f}, Margin: {best_match_score - second_best_score:.2f})")
            results.append((None, best_match_score))

    print("Annotating results...")
    # Step 4: Annotate the image
    annotated_image = image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    print("Face recognition completed.")
    return annotated_image, results

def numpy_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)  # Avoid division by zero

def recognize_faces_faiss(input_image, database, threshold=0.6, nlist=16, nprobe=4):
    """
    Optimized face recognition using FAISS with approximate nearest neighbor search.
    """
    print("Starting to recognize faces using FAISS.")

    # Step 1: Extract and crop faces from the input image
    boxes, faces = extract_face(input_image)  # Replace with your face detection function
    if not faces:
        return input_image, []

    print("Encoding faces.")
    # Step 2: Generate embeddings for the detected faces
    embeddings = encode_faces(faces, facenet)  # Replace with your face embedding model

    print("Preparing FAISS index with on-the-fly normalization.")
    # Step 3: Prepare the FAISS index with non-normalized database embeddings
    faiss.omp_set_num_threads(8)  # Adjust based on your system
    all_db_embeddings = []
    embedding_to_name_map = []

    # Flatten database and map embeddings to names
    for name, embedding_list in database.items():
        for embedding in embedding_list:
            all_db_embeddings.append(embedding)
            embedding_to_name_map.append(name)

    # Convert embeddings to a NumPy array
    all_db_embeddings = np.array(all_db_embeddings, dtype='float32')

    all_db_embeddings = numpy_normalize(all_db_embeddings)

    # Create and configure FAISS index
    dimension = all_db_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index = faiss.IndexIVFFlat(index, dimension, nlist)  # Inverted index
    # Use a subset of embeddings for training
    num_training_points = max(1000, len(all_db_embeddings) // 10)  # 10% or at least 1000 points
    training_embeddings = all_db_embeddings[np.random.choice(len(all_db_embeddings), num_training_points, replace=False)]

    # Train and add embeddings
    index.train(training_embeddings)
    index.add(all_db_embeddings)  # Add normalized embeddings
    index.nprobe = nprobe  # Set the number of clusters to search

    print("Normalizing and searching embeddings.")
    # Step 4: Normalize input embeddings and search using FAISS
    embeddings = np.array(embeddings, dtype='float32')
    embeddings = normalize(embeddings, axis=1, norm='l2')
    k = 1  # Top-1 match
    distances, indices = index.search(embeddings, k)

    print("Processing search results.")
    # Step 5: Process the FAISS results
    results = []
    for dist, idx in zip(distances[:, 0], indices[:, 0]):
        if dist > threshold and idx < len(embedding_to_name_map):
            results.append((embedding_to_name_map[idx], dist))  # Map embedding back to person's name
        else:
            results.append((None, dist))

    print("Annotating the image.")
    # Step 6: Annotate the image with the recognition results
    annotated_image = input_image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated_image, results


# KMeans clustering for grouping similar faces
def apply_kmeans_clustering(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_

# Scikit-learn NearestNeighbors for K-NN search
def sklearn_nn_search(database_embeddings, query_embeddings, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')  # Using cosine distance
    nn.fit(database_embeddings)
    distances, indices = nn.kneighbors(query_embeddings)
    
    return distances, indices

# Function to recognize and group faces
def recognize_and_group_faces(input_image, database, threshold=0.7, k=5, clustering=False):
    # Assuming embeddings for faces are pre-computed and loaded
    database_embeddings = np.array([db_emb for name, db_emb in database.items()])
    database_names = list(database.keys())
    
    # Step 1: Extract faces from the input image
    boxes, faces = extract_face(input_image)  # Replace with actual face extraction
    if not faces:
        print("No faces detected in the input image.")
        return input_image, []

    # Step 2: Generate embeddings for detected faces
    query_embeddings = encode_faces(faces, facenet)  # Replace with your embedding extraction

    # Normalize embeddings
    query_embeddings = numpy_normalize(query_embeddings)
    
    # Step 3: Perform nearest neighbor search
    distances, indices = sklearn_nn_search(database_embeddings, query_embeddings, k)

    # If clustering is enabled, group the query embeddings into clusters
    if clustering:
        labels = apply_kmeans_clustering(query_embeddings, n_clusters=3)
        print(f"Clustering results: {labels}")
    
    # Step 4: Process the search results and assign labels
    results = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        best_match_name = None
        best_match_score = -1

        for j, index in enumerate(idx):
            if dist[j] < threshold:
                best_match_name = database_names[index]
                best_match_score = dist[j]

        # Check if the best match score is above the threshold
        if best_match_name:
            print(f"Face {i+1}: Matched with {best_match_name} (Score: {best_match_score:.2f})")
            results.append((best_match_name, best_match_score))
        else:
            print(f"Face {i+1}: No reliable match found.")
            results.append((None, None))
    
    # Step 5: Annotate the image with the recognition results
    annotated_image = input_image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated_image, results
