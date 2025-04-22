"""Image preprocessing utilities for math recognition."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_image(image: np.ndarray, 
                    target_resolution: Tuple[int, int] = (256, 1024),
                    invert: bool = False, 
                    normalize: bool = True,
                    crop_margin: float = 0.05,
                    min_margin_pixels: int = 20) -> np.ndarray:
    """Preprocess image for the PosFormer model.
    
    Args:
        image: Input image as numpy array (grayscale)
        target_resolution: Target resolution (height, width)
        invert: Whether to invert colors (black becomes white, white becomes black)
        normalize: Whether to normalize pixel values to [0, 1]
        crop_margin: Margin to add around content after cropping (as % of dimensions)
        min_margin_pixels: Minimum margin in pixels
        
    Returns:
        Preprocessed image as numpy array [1, H, W, 1] ready for model input
    """
    # Ensure image is grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Invert colors if needed so content is black on white
    if np.mean(gray) < 127 and not invert:
        gray = 255 - gray
    elif invert:
        gray = 255 - gray
    
    # Binarize the image to help find content
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find content bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the bounding box that contains all contours
    if contours:
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        # Add margin
        margin_x = max(int(crop_margin * gray.shape[1]), min_margin_pixels)
        margin_y = max(int(crop_margin * gray.shape[0]), min_margin_pixels)
        
        min_x = max(0, min_x - margin_x)
        min_y = max(0, min_y - margin_y)
        max_x = min(gray.shape[1], max_x + margin_x)
        max_y = min(gray.shape[0], max_y + margin_y)
        
        # Crop the image
        cropped = gray[min_y:max_y, min_x:max_x]
    else:
        # No content found, use the original image
        cropped = gray
    
    # Resize to target resolution
    resized = resize_with_aspect_ratio(cropped, target_resolution)
    
    # Make sure the image is the exact target size by padding if necessary
    target_h, target_w = target_resolution
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # Center the resized image in the target space
    h, w = resized.shape
    y_offset = (target_h - h) // 2
    x_offset = (target_w - w) // 2
    
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = resized
    
    # Normalize to [0, 1] if requested
    if normalize:
        padded = padded.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions [B, H, W, C]
    result = padded.reshape(1, target_h, target_w, 1)
    
    return result


def resize_with_aspect_ratio(image: np.ndarray, 
                            target_size: Tuple[int, int], 
                            pad_with_zeros: bool = False) -> np.ndarray:
    """Resize an image maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        pad_with_zeros: Whether to pad with zeros to reach exact target size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate the scaling factors for both dimensions
    scale_h = target_h / h
    scale_w = target_w / w
    
    # Use the smaller scaling factor to maintain aspect ratio
    scale = min(scale_h, scale_w)
    
    # Calculate new dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if pad_with_zeros:
        # Create a black canvas of the target size
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Calculate the position to place the resized image (center)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place the resized image in the center of the canvas
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    return resized