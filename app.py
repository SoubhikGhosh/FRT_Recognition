from flask import Flask, request, jsonify
from flask_cors import cross_origin
import cv2
import base64
from utilities.dbUtils import add_face_to_db, build_face_database
from utilities.faceUtils import encode_faces, extract_face, preprocess_image
from utilities.searchUtil import run_ann_search
from utilities.testDbConnection import test_connection
import numpy as np
from bootstrapProperties import PREPROCESS

app = Flask(__name__)

@app.route('/test-db-connection', methods=['GET'])
@cross_origin(origins=['*'])
def test_db_connection():
    if test_connection() == "Connection successful!":
        return jsonify(test_connection()), 200
    else:
        return jsonify(test_connection()), 500
    
@app.route('/load-database', methods=['POST'])
@cross_origin(origins=['*'])
def load_data_to_postgres():
    try:
        # Parse the JSON request
        data = request.get_json()

        # Check if the 'folder' are present
        if 'folder' not in data:
            return jsonify({"error": "folder name missing"}), 400
        
        IMAGE_FOLDER = data['folder']
        build_face_database(IMAGE_FOLDER)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/register', methods=['POST'])
@cross_origin(origins=['*'])
def add_face():
    """
    Endpoint to add a face to the database. 
    Accepts JSON with the key "name" and "image" (base64).
    Converts the face to an embedding and stores it in the database.
    """
    try:
        # Parse the JSON request
        data = request.get_json()

        # Check if the 'name' and 'image' keys are present
        if 'name' not in data or 'image' not in data:
            return jsonify({"error": "Name or image data missing"}), 400
        
        name = data['name']
        img_data = data['image']
        
        # If the image string includes data:image prefix, strip it
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]

        # Decode the base64 string to bytes
        img_bytes = base64.b64decode(img_data)

        # Convert the bytes to an OpenCV image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if (PREPROCESS):
            image = preprocess_image(image)
        # Extract faces from the image
        _, faces = extract_face(image)
        if not faces:
            return jsonify({"message": "No face detected"}), 400

        # Encode the detected faces using FaceNet
        embeddings = encode_faces(faces)
        embedding = embeddings[0]  # Assuming one face per image

        # Store the face embedding and name in the database
        add_face_to_db(name, embedding)
        print("Face added successfully")
        return jsonify({"message": "Face added successfully to the database"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recognize', methods=['POST'])
@cross_origin(origins=['*'])
def recognize_face():
    """
    Endpoint to recognize a face from a base64-encoded image.
    Accepts JSON with the key "image" containing the base64 string.
    Returns recognition results.
    """
    try:
        # Parse the JSON request
        data = request.get_json()

        # Check if the 'image' key is present
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get the base64-encoded image string and decode it
        img_data = data['image']
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]

        # Decode the base64 string to bytes
        img_bytes = base64.b64decode(img_data)

        # Convert the bytes to an OpenCV image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if (PREPROCESS):
            image = preprocess_image(image)
        # Extract faces from the image
        _, faces = extract_face(image)
        if not faces:
            return jsonify({"message": "No face detected"}), 400

        # Encode the detected faces using FaceNet
        embeddings = encode_faces(faces)
        embedding = embeddings[0]  # Assuming one face per image

        # Perform an ANN search on the embeddings stored in the database
        results = run_ann_search(embedding)

        # Check for specific error messages from ANN search and return appropriate responses
        if "message" in results and results["message"] == "Face detected but not recognized":
            return jsonify({"message": "Face detected but not recognized"}), 400
        if "message" in results and results["message"] == "No matching faces found":
            return jsonify({"message": "No matching faces found"}), 400
        print(results)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=False)
