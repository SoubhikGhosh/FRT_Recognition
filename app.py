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

def generate_frames():
    """ Generate video frames for streaming """

    # Initialize camera
    camera = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with your video source
    while True:
        success, frame = camera.read()
        if not success:
            print("unsuccessful camera read")
            break

        # Detect and recognize faces in the frame
        annotated_frame, results = recognize_faces(frame, database)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        if not _:
            print("Error encoding frame")
            continue

        # Yield the frame for video streaming
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """ 
    Route to stream video to the HTML page 
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_faces')
def get_faces():
    """ 
    Route to fetch face recognition data as JSON 
    """
    # Initialize camera
    camera2 = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with your video source
    success, frame = camera2.read()
    if not success or frame is None or frame.size == 0:
        # Handle case where frame is empty or invalid
        return jsonify({"faces": []})

    # Recognize faces in the current frame
    _, results = recognize_faces(frame, database)

    # Prepare face data for the response
    face_data = [
        {
            "name": name,
            "confidence": convert_to_float(score)  # Convert NumPy float32 to Python float
        }
        for (name, score) in results
    ]

    return jsonify({"faces": face_data})

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

@app.route('/')
def index():
    """ Route to render the HTML page """
    return render_template('index.html')

if __name__ == "__main__":
    # Ensure the face database exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Face database folder '{IMAGE_FOLDER}' does not exist.")
        exit(1)

    app.run(host="0.0.0.0", port=5002, debug=True)