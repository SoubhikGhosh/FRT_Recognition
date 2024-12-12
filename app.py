from flask import Flask, request, render_template, Response, jsonify
import cv2
import os
from utils import recognize_faces, build_face_database
import numpy as np
import base64

app = Flask(__name__)

# Load the face database
IMAGE_FOLDER = "known_faces"  # Path to the folder containing images of known faces
database = build_face_database(IMAGE_FOLDER)

def convert_to_float(value):
    """ Convert NumPy float32 to Python native float """
    if isinstance(value, np.float32):
        return float(value)
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(v) for v in value]
    return value

@app.route('/recognize_face', methods=['POST'])
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
        np_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Perform face recognition
        _, results = recognize_faces(frame, database)

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

    app.run(port=5002, debug=True)