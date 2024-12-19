import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
from utils import get_next_filename

# Function for rotation
def rotate_image(image, angle_range=(-30, 30)):
    angle = random.uniform(angle_range[0], angle_range[1])
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Function for translation
def translate_image(image, tx_range=(-30, 30), ty_range=(-30, 30)):
    tx = random.uniform(tx_range[0], tx_range[1])
    ty = random.uniform(ty_range[0], ty_range[1])
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    rows, cols, _ = image.shape
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

# Function for scaling
def scale_image(image, scale_range=(0.8, 1.2)):
    scale = random.uniform(scale_range[0], scale_range[1])
    rows, cols, _ = image.shape
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
    return scaled_image

# Function for shearing
def shear_image(image, shear_range=(-0.2, 0.2)):
    shear = random.uniform(shear_range[0], shear_range[1])
    rows, cols, _ = image.shape
    M = np.float32([[1, shear, 0], [shear, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    return sheared_image

# Function for horizontal flipping
def flip_image(image):
    return cv2.flip(image, 1)

# Function for cropping
def crop_image(image, crop_fraction=0.1):
    h, w, _ = image.shape
    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)
    x1 = random.randint(0, crop_w)
    y1 = random.randint(0, crop_h)
    x2 = w - random.randint(0, crop_w)
    y2 = h - random.randint(0, crop_h)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image
# Function for brightness adjustment
def adjust_brightness(image, factor_range=(0.5, 1.5)):
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    return np.array(enhancer.enhance(factor))

# Function for contrast adjustment
def adjust_contrast(image, factor_range=(0.5, 1.5)):
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    return np.array(enhancer.enhance(factor))

# Function for saturation adjustment
def adjust_saturation(image, factor_range=(0.5, 1.5)):
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Color(Image.fromarray(image))
    return np.array(enhancer.enhance(factor))

# Function for hue shifting
def adjust_hue(image, shift_range=(-10, 10)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 0] = np.clip(hsv_image[..., 0] + random.randint(shift_range[0], shift_range[1]), 0, 179)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Function for gamma correction
def adjust_gamma(image, gamma_range=(0.5, 2.0)):
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lookup_table)
# Function for adding Gaussian noise
def add_gaussian_noise(image, mean=0, var=0.1):
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

# Function for adding salt-and-pepper noise
def add_salt_pepper_noise(image, amount=0.01):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(total_pixels * amount)
    num_pepper = int(total_pixels * amount)
    # Salt noise (white pixels)
    for _ in range(num_salt):
        x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        noisy_image[x, y] = 255
    # Pepper noise (black pixels)
    for _ in range(num_pepper):
        x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        noisy_image[x, y] = 0
    return noisy_image
def elastic_deform(image, alpha=34, sigma=4):
    """
    Elastic deformation for image augmentation.
    Args:
        image (ndarray): Input image.
        alpha (float): Scaling factor for deformation intensity.
        sigma (float): Standard deviation for Gaussian filter (controls smoothness of deformation).
    
    Returns:
        ndarray: Elastic deformed image.
    """
    # Get the shape of the image
    shape = image.shape
    height, width = shape[:2]
    
    # Generate random displacement fields
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0)
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0)
    
    # Scale the displacement fields
    dx *= alpha
    dy *= alpha
    
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply the displacement to the meshgrid
    distorted_image = np.zeros_like(image)
    for i in range(3):  # For each color channel (R, G, B)
        distorted_image[:, :, i] = map_coordinates(image[:, :, i], [y + dy, x + dx], order=1, mode='reflect')
    
    return distorted_image
# Function for Gaussian blur
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function for Motion blur
def apply_motion_blur(image, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

# Function for adding random occlusions
def add_occlusions(image, num_occlusions=1, size_range=(20, 60)):
    image_copy = image.copy()
    h, w, _ = image.shape
    for _ in range(num_occlusions):
        occlusion_size = random.randint(size_range[0], size_range[1])
        x1 = random.randint(0, w - occlusion_size)
        y1 = random.randint(0, h - occlusion_size)
        x2, y2 = x1 + occlusion_size, y1 + occlusion_size
        image_copy[y1:y2, x1:x2] = 0  # Black occlusion
    return image_copy

def save_augmentations(image_path, save_dir):

    image = cv2.imread(image_path)

    rotated_image = rotate_image(image)
    translated_image = translate_image(image)
    scaled_image = scale_image(image)
    flipped_image = flip_image(image)
    brightness_image = adjust_brightness(image)
    noisy_image = add_gaussian_noise(image)
    blurred_image = apply_gaussian_blur(image)
    occluded_image = add_occlusions(image)
    elastic_image = elastic_deform(image)
    motion_blurred_image = apply_motion_blur(image)
    contrast_image = adjust_contrast(image)
    saturation_image = adjust_saturation(image)
    hue_image = adjust_hue(image)
    gamma_image = adjust_gamma(image)
    sheared_image = shear_image(image)
    cropped_image = crop_image(image)
    combined_image = adjust_brightness(adjust_contrast(image))

    augmentations = [rotated_image, translated_image, scaled_image, flipped_image, brightness_image, noisy_image, 
                     blurred_image, occluded_image, elastic_image, motion_blurred_image, contrast_image, saturation_image,
                     hue_image, gamma_image, sheared_image, cropped_image, combined_image]
    return augmentations
        


