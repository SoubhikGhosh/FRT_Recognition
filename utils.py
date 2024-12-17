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

def save_and_process_image(name, base64_image, save_dir, database):
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
        file_count = len([f for f in os.listdir(user_dir) if f.endswith(('.jpg', '.png'))])
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

        # Detect and encode faces
        _, faces = extract_face(image)
        if not faces:
            return {"error": "No faces detected in the image"}

        embeddings = encode_faces(faces, facenet)

        # Update the database
        if name not in database:
            print("name not in database, creating record.")
            database[name] = []
        database[name].extend(embeddings)

        # Save updated database to .pkl
        save_database(database)
        print("updated .pkl file with the new entry.")

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
    embeddings = []
    for face in faces:
        face = torch.tensor(face.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        embedding = facenet(face).detach().cpu().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

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
    save_database(database)
    return database

def recognize_faces(image, database, threshold=0.5):
    """
    Detect and recognize faces in an image by comparing embeddings 
    with all known embeddings in the database.
    """

    print("starting to recognize faces.")
    # Step 1: Detect faces in the image
    boxes, faces = extract_face(image)
    if not faces:
        print("no faces detected.")
        return image, []
    
    print("encoding faces.")
    # Step 2: Encode detected faces into embeddings
    embeddings = encode_faces(faces, facenet)

    results = []

    print("matching faces.")
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

    print("annotating results.")
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

    print("starting to recognize faces using FAISS.")
    # Step 1: Extract and crop faces from the input image
    boxes, faces = extract_face(input_image)
    if not faces:
        return input_image, []

    print("encoding faces.")
    # Step 2: Generate embeddings for the detected faces
    embeddings = encode_faces(faces, facenet)


    print("matching faces.")
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

    print("annotating results.")
    # Step 6: Annotate the image with the recognition results
    annotated_image = input_image.copy()
    for (box, (name, score)) in zip(boxes, results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated_image, results
