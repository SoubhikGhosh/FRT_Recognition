import mediapipe as mp
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MediaPipe BlazeFace
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_face(image, target_size=(160, 160)):
    """
    Extract faces using MediaPipe BlazeFace.
    
    Parameters:
    - image: The image to process (OpenCV format).
    - target_size: The target size to resize detected faces.
    
    Returns:
    - boxes: The bounding boxes of detected faces.
    - cropped_faces: A list of cropped face images.
    """
    # Convert the image to RGB as MediaPipe expects RGB input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize the face detection model
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        # Perform face detection
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None, None
        
        boxes = []
        cropped_faces = []

        # Extract faces from detected bounding boxes
        for detection in results.detections:
            # Get the bounding box for each detected face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            x2 = int((bboxC.xmin + bboxC.width) * iw)
            y2 = int((bboxC.ymin + bboxC.height) * ih)
            boxes.append([x1, y1, x2, y2])

            # Crop the face from the image
            face = image[y1:y2, x1:x2]
            if face.size == 0:  # Ensure face is valid
                continue
            face_resized = cv2.resize(face, target_size)
            cropped_faces.append(face_resized)

        cv2.destroyAllWindows()  # If any windows were created during processing
        return boxes, cropped_faces

def encode_faces(faces):
    embeddings = []
    for face in faces:
        face = torch.tensor(face.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        embedding = facenet(face).detach().cpu().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened