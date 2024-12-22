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

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction to an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def color_normalization(image):
    """Normalize the image to have consistent color channels."""
    # Convert to LAB color space for better illumination handling
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE on the L channel to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back the LAB channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    normalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized_image

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, gamma=1.0):
    """Return a sharpened version of the image, using an unsharp mask with CLAHE,
      gamma correction, and color normalization."""
    
    # Step 1: Apply CLAHE and color normalization
    image = color_normalization(image)
    
    # Step 2: Apply Gamma Correction
    image = gamma_correction(image, gamma)
    
    # Step 3: Apply Unsharp Mask
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    return sharpened

def align_face(image):
    """
    Align face using MediaPipe face landmarks.
    """
    # Convert image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to find face landmarks
    results = face_mesh.process(rgb_image)

    # Check if any face is detected
    if not results.multi_face_landmarks:
        raise ValueError("No faces detected")

    # Select the first face
    face_landmarks = results.multi_face_landmarks[0]
    
    # Get the bounding box of the face (simplified method using landmarks)
    height, width, _ = image.shape
    x_min, y_min = int(width), int(height)
    x_max, y_max = 0, 0

    for landmark in face_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    
    # Crop the face region using the bounding box
    face = image[y_min:y_max, x_min:x_max]
    
    # Resize the face to the expected input size for InceptionResNetV1
    return cv2.resize(face, (160, 160))  # Resize to 160x160

def preprocess_image(image):
    """Preprocess the image before passing it to InceptionResNetV1."""
    
    # Step 1: Align the face using MediaPipe
    image = align_face(image)
    
    # Step 2: Preprocess (sharpening, color normalization, gamma correction)
    image = unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, gamma=1.0)

    # Step 3: Normalize the image by scaling to [0, 1] and performing mean normalization
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0  # Scale to [0, 1]
    
    # If needed, perform additional normalization based on pre-trained model
    # For VGGFace2: mean values are [93.5940, 104.7624, 129.1863]
    image = image - np.array([93.5940, 104.7624, 129.1863]) / 255.0  # Subtract mean (based on VGGFace2 training)


    # Step 4: Expand dimensions to match the model input (160x160, 3 channels)
    image = np.expand_dims(image, axis=0)
    
    return image
