import psycopg2
from tqdm import tqdm
import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from utilities.faceUtils import extract_face, encode_faces
from utilities.testDbConnection import DB_CONFIG

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def connect_to_db():
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

def save_embedding_to_db(person_name, embedding):
    """Save an embedding to the PostgreSQL database."""
    conn = connect_to_db()
    cursor = conn.cursor()

    # Insert embedding
    sql = """
    INSERT INTO face_embeddings (person_name, embedding)
    VALUES (%s, %s);
    """
    cursor.execute(sql, (person_name, embedding.tolist()))
    conn.commit()
    cursor.close()
    conn.close()

def build_face_database(image_folder):
    """
    Builds a face database by extracting face embeddings from all images 
    in subdirectories named after individuals, storing results in PostgreSQL.
    """
    # Gather all image paths
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in ['.jpg', '.png', '.jpeg']:  # Only consider image files
                image_paths.append(os.path.join(root, file))

    # Process images with a progress bar
    with tqdm(total=len(image_paths), desc="Loading database", ncols=100, unit="file") as pbar:
        for image_path in image_paths:
            person_name = os.path.basename(os.path.dirname(image_path))
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                pbar.update(1)
                continue

            boxes, faces = extract_face(image)  # Assume this function exists
            if faces:
                embeddings = encode_faces(faces, facenet)  # Assume this function exists
                for embedding in embeddings:
                    save_embedding_to_db(person_name, embedding)
            else:
                print(f"Face not detected in {image_path}")
            
            pbar.update(1)

    print("Database loaded into PostgreSQL.")

def add_face_to_db(name, embedding):
    """Store a face's name and embedding in the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Convert embedding to the pgvector format
        embedding_str = f"[{','.join(map(str, embedding))}]"

        # SQL query to insert the face into the database
        sql = """
        INSERT INTO face_embeddings (person_name, embedding)
        VALUES (%s, %s);
        """
        cursor.execute(sql, (name, embedding_str))
        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error storing face data: {e}")
        raise