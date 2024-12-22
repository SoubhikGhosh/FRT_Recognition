from utilities.testDbConnection import DB_CONFIG
import psycopg2
import numpy as np

# Set your threshold for confidence (e.g., 0.7)
CONFIDENCE_THRESHOLD = 0.7

def run_ann_search(query_embedding):
    """Perform ANN search using the query embedding in PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Ensure query_embedding is in the correct format (numpy array)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()  # Convert to list if it's a numpy array

    # Convert the list to the pgvector type by using the 'vector' type in PostgreSQL
    query_embedding_pgvector = f'[{",".join(map(str, query_embedding))}]'

    # SQL query to find the top 1 most similar embedding
    sql = """
    SELECT person_name, embedding, 
           (embedding <=> %s) AS distance  
    FROM face_embeddings
    ORDER BY distance  
    LIMIT 1;
    """
    
    # Pass the query embedding as a pgvector formatted string
    cursor.execute(sql, (query_embedding_pgvector,))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    # Prepare results in a more readable format
    if results:
        person_name, _, distance = results[0]
        confidence = 1.0 - float(distance)
        print(f"detected: {person_name} confidence: {confidence}")
        
        # Check if confidence is below the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return {"message": "Face detected but not recognized"}
        
        return {"name": person_name, "confidence": confidence}
    else:
        return {"message": "No matching faces found"}