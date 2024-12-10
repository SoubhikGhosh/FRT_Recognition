import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss



def recognize_faces_video(video_path, database, mtcnn, facenet, output_path, threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video source.")
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_image, _ = recognize_faces_faiss(frame, database, mtcnn, facenet, threshold)
        out.write(annotated_image)

    cap.release()
    out.release()

# Face recognition model (FaceNet pre-trained)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face(image, mtcnn, target_size=(160, 160)):
    """
    Detects and crops the face from the image.
    """
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None
    cropped_faces = []
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        face = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face, target_size)
        cropped_faces.append(face_resized)
    return cropped_faces

def encode_faces(faces, facenet):
    """
    Generates embeddings for detected faces.
    """
    embeddings = []
    for face in faces:
        # Convert to PyTorch tensor
        face = torch.tensor(face.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        embedding = facenet(face).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

def build_face_database(image_folder, mtcnn, facenet):
    """
    Creates a database of known faces from images in the given folder.
    """
    database = {}
    for file in os.listdir(image_folder):
        name, ext = os.path.splitext(file)
        if ext.lower() not in ['.jpg', '.png']:
            continue
        image = cv2.imread(os.path.join(image_folder, file))
        faces = extract_face(image, mtcnn)
        if faces:
            embeddings = encode_faces(faces, facenet)
            database[name] = embeddings[0]
    return database

def recognize_faces_faiss(input_image, database, mtcnn, facenet, threshold):
    """
    Recognizes faces in the input image by comparing with the database using FAISS.
    
    Parameters:
    - input_image: The image to process.
    - database: A dictionary of precomputed embeddings (name -> embedding).
    - mtcnn: The face detector.
    - facenet: The face recognition model.
    - threshold: Similarity threshold for recognition.
    
    Returns:
    - Annotated image with recognition results.
    """
    # Extract and crop faces
    faces = extract_face(input_image, mtcnn)
    if not faces:
        return input_image, []

    # Generate embeddings for detected faces
    embeddings = encode_faces(faces, facenet)
    
    # Prepare FAISS index
    # Convert database to a format suitable for FAISS
    db_embeddings = np.array(list(database.values())).astype('float32')
    db_names = list(database.keys())
    
    # Create FAISS index for L2 similarity (use IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatIP(db_embeddings.shape[1])  # Inner Product for cosine similarity
    faiss.normalize_L2(db_embeddings)  # Normalize embeddings for cosine similarity
    index.add(db_embeddings)

    
    # Normalize face embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Search database for nearest matches
    distances, indices = index.search(embeddings, 1)  # Top-1 match for each embedding

    results = []
    for dist, idx in zip(distances[:, 0], indices[:, 0]):
        if dist > threshold and idx < len(db_names):
            results.append((db_names[idx], dist))
        else:
            results.append((None, dist))
    
    # Annotate image
    annotated_image = input_image.copy()
    for (box, (name, score)) in zip(mtcnn.detect(input_image)[0], results):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = f"{name} ({score:.2f})" if name else "Unknown"
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
    
    return annotated_image, results