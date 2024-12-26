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

            _, faces = extract_face(image)  # Assume this function exists
            if faces:
                embeddings = encode_faces(faces, facenet)  # Assume this function exists
                for embedding in embeddings:
                    save_embedding_to_db(person_name, embedding)
            else:
                print(f"Face not detected in {image_path}")
            
            pbar.update(1)

    print("Database loaded into PostgreSQL.")

def add_face_to_db(phone_number, embedding, name):
    """
    Store a face's phone number, name, and embedding in the database.
    If the name is not passed, retrieve it from the database using the phone number.
    Ensures phone number and embedding are stored in their respective tables.
    Handles duplicates for phone numbers gracefully.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # If name is not provided, query the database to get the name based on phone_number
        if not name:
            sql_get_name = """
            SELECT person_name 
            FROM person_phone_mapping 
            WHERE phone_number = %s;
            """
            cursor.execute(sql_get_name, (phone_number,))
            result = cursor.fetchone()
            if result:
                name = result[0]  # If a name is found, use it
            else:
                print(f"Error: No name found for phone number {phone_number}.")
                return  # Exit if no name found for the given phone number

        # Convert embedding to the pgvector format
        embedding_str = f"[{','.join(map(str, embedding))}]"

        # Try to insert into person_phone_mapping (phone number and name)
        try:
            sql_person_phone = """
            INSERT INTO person_phone_mapping (person_name, phone_number)
            VALUES (%s, %s);
            """
            cursor.execute(sql_person_phone, (name, phone_number))
        except psycopg2.errors.UniqueViolation:
            print(f"Phone number {phone_number} already exists in person_phone_mapping.")

        # Insert the embedding and phone number into face_embeddings
        sql_face_embedding = """
        INSERT INTO face_embeddings (phone_number, embedding)
        VALUES (%s, %s);
        """
        cursor.execute(sql_face_embedding, (phone_number, embedding_str))

        # Commit the transaction
        conn.commit()

    except Exception as e:
        print(f"Error storing face data: {e}")
        raise

    finally:
        cursor.close()
        conn.close()

def insert_feedback (embedding, actual_phone_number, predicted_phone_number, confidence_score, feedback_type):
    conn = connect_to_db()
    cursor = conn.cursor()

    try:

        # Insert feedback record
        cursor.execute("""
            INSERT INTO feedback (actual_phone_number, predicted_phone_number, confidence_score, embedding, feedback_type)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (actual_phone_number, predicted_phone_number, confidence_score, embedding, feedback_type))
        
        conn.commit()
        print("Feedback successfully inserted.")
    
    except Exception as e:
        print(f"Error inserting feedback: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()

