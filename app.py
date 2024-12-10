from flask import Flask, request, jsonify, send_file
import cv2
import os
from models import mtcnn, facenet, build_face_database, recognize_faces_video

app = Flask(__name__)

# Load the face database
DATABASE_FOLDER = "known_faces"
database = build_face_database(DATABASE_FOLDER, mtcnn, facenet)

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        # Check if a video file was uploaded
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({"error": "No video file provided"}), 400
        
        # Save uploaded video
        input_video_path = os.path.join("static", "input.mp4")
        output_video_path = os.path.join("static", "output.mp4")
        video_file.save(input_video_path)
        
        # Process the video
        recognize_faces_video(input_video_path, database, mtcnn, facenet, output_video_path)
        
        # Return the processed video
        return send_file(output_video_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(host='0.0.0.0', port=5000, debug=True)
