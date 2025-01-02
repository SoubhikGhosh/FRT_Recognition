from flask import Flask, request, jsonify
from flask_cors import cross_origin
import cv2
import base64
from utilities.dbUtils import add_face_to_db, build_face_database, insert_feedback, update_phone_number_in_db, check_if_registered
from utilities.faceUtils import encode_faces, extract_face, unsharp_mask
from utilities.searchUtil import run_ann_search
from utilities.testDbConnection import test_connection
import numpy as np

app = Flask(__name__)

# Load the face database (can be uncommented if needed in the future)
# IMAGE_FOLDER = "known_faces"
# database = build_face_database(IMAGE_FOLDER)

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
    Accepts JSON with the key "name", "image" (base64), and "phone_number".
    Converts the face to an embedding and stores it in the database.
    """
    try:
        # Parse the JSON request
        data = request.get_json()

        # Define the required keys
        required_keys = ['name', 'image', 'phone_number']

        # Check if all required keys are present
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": f"Missing keys: {', '.join(missing_keys)}"}), 400

        name = data['name']
        img_data = data['image']
        phone_number = data['phone_number']

        # Validate phone number length and format
        if not phone_number.isdigit() or len(phone_number) != 10:
            return jsonify({"error": "Phone number must be exactly 10 digits"}), 400

        # Format the phone number to xxx-xxx-xxxx
        formatted_phone_number = f"{phone_number[:3]}-{phone_number[3:6]}-{phone_number[6:]}"
        phone_number = formatted_phone_number  # Update the variable with the formatted value

        # If the image string includes a data:image prefix, strip it
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]

        # Decode the base64 string to bytes
        try:
            img_bytes = base64.b64decode(img_data)
        except Exception as e:
            return jsonify({"error": "Invalid image data, unable to decode base64."}), 400

        # Convert the bytes to an OpenCV image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image. Ensure the image is valid."}), 400

        image = unsharp_mask(image)

        # Extract faces from the image
        _, faces = extract_face(image)
        if not faces:
            return jsonify({"message": "No face detected."}), 400

        # Encode the detected faces using FaceNet
        embeddings = encode_faces(faces)
        embedding = embeddings[0]  # Assuming one face per image

        # Store the face embedding and name in the database
        try:
            add_face_to_db(phone_number, embedding, name)
        except Exception as e:
            return jsonify({"error": f"Error storing face data: {str(e)}"}), 500

        print("Face added successfully")
        return jsonify({"message": "Face added successfully to the database"}), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

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

        image = unsharp_mask(image)
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

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
@cross_origin(origins=['*'])
def register_feedback_endpoint():
    """
    Endpoint to receive feedback on face recognition results.
    """
    try:
        # Parse the JSON request
        data = request.get_json()

        # Define the required keys
        required_keys = ['predicted_phone_number', 'confidence_score', 'image', 'feedback_type']

        # Check if all required keys are present
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": f"Missing keys: {', '.join(missing_keys)}"}), 400
        
        # Extract values from the request
        img_data = data.get('image')
        predicted_phone_number = data.get('predicted_phone_number')
        confidence_score = data.get('confidence_score')
        feedback_type = data.get('feedback_type')

        if feedback_type.lower() == "correct" and 'actual_phone_number' not in data:
            actual_phone_number = data.get('predicted_phone_number')
        else:
            actual_phone_number = data.get('actual_phone_number')

            name =  check_if_registered(actual_phone_number)

            if not name:
                return jsonify({"error": "Please register your phone number before giving feedback."}), 400



            # Validate phone number length and format
            if not len(actual_phone_number) != 10:
                return jsonify({"error": "Phone number must be exactly 10 digits"}), 400

            # Format the phone number to xxx-xxx-xxxx
            formatted_phone_number = f"{actual_phone_number[:3]}-{actual_phone_number[3:6]}-{actual_phone_number[6:]}"
            actual_phone_number = formatted_phone_number  # Update the variable with the formatted value

        # Check if image data is provided
        if not img_data:
            return jsonify({"error": "Image data is required"}), 400

        # If the image string includes 'data:image' prefix, strip it
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]

        # Decode the base64 string to bytes
        try:
            img_bytes = base64.b64decode(img_data)
        except Exception as e:
            return jsonify({"error": "Invalid image format. Could not decode the image."}), 400

        # Convert the bytes to an OpenCV image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # If the image is invalid (empty or cannot be processed), return an error
        if image is None:
            return jsonify({"error": "Failed to decode the image. Invalid image."}), 400

        # Apply unsharp mask to enhance the image (if required)
        image = unsharp_mask(image)

        # Extract faces from the image
        _, faces = extract_face(image)
        if not faces:
            return jsonify({"error": "No face detected in the image"}), 400

        # Encode the detected faces using FaceNet
        embeddings = encode_faces(faces)
        embedding = embeddings[0]  # Assuming one face per image

        # If feedback type is "correct", add the face to the database
        if feedback_type.lower() == "correct":
            try:
                add_face_to_db(actual_phone_number, embedding, None)
            except Exception as e:
                return jsonify({"error": f"Error storing face data: {str(e)}"}), 500

        # Insert the feedback in the database
        try:
            insert_feedback(embedding, actual_phone_number, predicted_phone_number, confidence_score, feedback_type)
        except Exception as e:
            return jsonify({"error": f"Error inserting feedback: {str(e)}"}), 500

        return jsonify({"message": "Feedback received successfully."}), 200

    except Exception as e:
        # Catch all other exceptions and return a 500 error
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/update-phone-number', methods=['POST'])
@cross_origin(origins=['*'])
def update_phone_number():
    """
    API endpoint to update a phone number in the database.
    Accepts JSON input with 'name' and 'new_phone_number'.
    """
    try:
        # Parse the JSON request
        data = request.get_json()

        # Validate input
        if not data or 'name' not in data or 'new_phone_number' not in data:
            return jsonify({"error": "Both 'name' and 'new_phone_number' are required"}), 400

        name = data['name']
        new_phone_number = data['new_phone_number']

        # Validate the new phone number
        if len(new_phone_number) != 10:
            return jsonify({"error": "Phone number must be exactly 10 digits"}), 400

        # Format the phone number to xxx-xxx-xxxx
        formatted_phone_number = f"{new_phone_number[:3]}-{new_phone_number[3:6]}-{new_phone_number[6:]}"
        new_phone_number = formatted_phone_number  # Update variable with formatted value

        # Call the utility function to update the phone number in the database
        result = update_phone_number_in_db(name, new_phone_number)

        # Handle responses based on utility results
        if result == "NAME_NOT_FOUND":
            return jsonify({"error": f"No record found for the name '{name}'"}), 404
        elif result == "PHONE_NUMBER_EXISTS":
            return jsonify({"error": f"The phone number '{new_phone_number}' is already assigned to another person"}), 409
        elif result == "SUCCESS":
            return jsonify({"message": "Phone number updated successfully"}), 200
        else:
            return jsonify({"error": "Unknown error occurred"}), 500

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=False)
