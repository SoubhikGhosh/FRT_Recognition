from flask import Flask, request, render_template, Response, jsonify
from flask_cors import CORS, cross_origin
import cv2
import os
from utils import recognize_faces, build_face_database, recognize_faces_faiss, convert_to_float, save_and_process_image, recognize_and_group_faces
import numpy as np
import base64
import time

app = Flask(__name__)

# Load the face database
IMAGE_FOLDER = "known_faces"  # Path to the folder containing images of known faces
database = build_face_database(IMAGE_FOLDER)

@app.route('/capture', methods=['POST'])
@cross_origin(origins=['*'])
def capture_image():
    """
    Endpoint to save an image, detect and encode faces, and update the database.
    """
    try:
        # Parse the request
        data = request.json
        if not data or 'name' not in data or 'image' not in data:
            return jsonify({"error": "Name and image data are required"}), 400
        
        name = data['name']
        base64_image = data['image']

        # Process the image and update the database
        result = save_and_process_image(name, base64_image, IMAGE_FOLDER, database)

        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        return jsonify({"message": result["message"], "file_path": result["file_path"]}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


@app.route('/recognize_face', methods=['POST'])
@cross_origin(origins=['*'])
def recognize_base64():
    """
    Endpoint to perform face recognition on a base64-encoded image.
    Accepts JSON with the key "image" containing the base64 string.
    Returns face recognition results as JSON.
    """
    try:
        # Parse the JSON request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        

        # Decode the base64 image
        image_data = base64.b64decode(data['image'].split(",")[1])
        # Get the current time as a float (seconds since epoch)
        current_time = time.time()

        # Convert to a time struct
        time_struct = time.localtime(current_time)

        # Format the time as a string (e.g., "YYYY-MM-DD HH:MM:SS")
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)

        file_path =  "recognise/"+time_str+".jpeg"
        with open(file_path, "wb") as f:
            f.write(image_data)
        print(f"Image saved at {file_path}")

        np_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Perform face recognition
        # Measure time for original recognize_faces
        start_time_original = time.time()
        _, results = recognize_and_group_faces(frame, database)
        time_original = time.time() - start_time_original
        print(f"time taken for normal cosine sim: {time_original}")


        # Prepare the response data
        face_data = [
            {
                "name": name,
                "confidence": convert_to_float(score)  # Convert NumPy float32 to Python float
            }
            for (name, score) in results
        ]

        print(f"face_data: {face_data}")
        return jsonify({"faces": face_data})

    except Exception as e:
        print(f"Error processing the base64 image: {e}")
        return jsonify({"error": "An error occurred during processing"}), 500

@app.route('/recognize_face_faiss', methods=['POST'])
@cross_origin(origins=['http://localhost:3000'])
def recognize_base64_faiss():
    """
    Endpoint to perform face recognition on a base64-encoded image.
    Accepts JSON with the key "image" containing the base64 string.
    Returns face recognition results as JSON.
    """
    try:
        # Parse the JSON request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 image
        image_data = base64.b64decode(data['image'].split(",")[1])
        np_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Perform face recognition
        # Measure time for FAISS-based recognize_faces_faiss
        start_time_faiss = time.time()
        _, results = recognize_faces_faiss(frame, database)
        time_faiss = time.time() - start_time_faiss

        print(f"time taken for faiss: {time_faiss}")

        # Prepare the response data
        face_data = [
            {
                "name": name,
                "confidence": convert_to_float(score)  # Convert NumPy float32 to Python float
            }
            for (name, score) in results
        ]

        return jsonify({"faces": face_data})

    except Exception as e:
        print(f"Error processing the base64 image: {e}")
        return jsonify({"error": "An error occurred during processing"}), 500

if __name__ == "__main__":
    # Ensure the face database exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Face database folder '{IMAGE_FOLDER}' does not exist.")
        exit(1)
    app.run(port=5000, debug=False)