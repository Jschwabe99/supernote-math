"""Detection of math regions in handwritten notes using OpenCV."""

import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathRegionDetector:
    """Detector for math regions in handwritten notes using OpenCV."""
    
    def __init__(self, model_path: Path = None):
        """Initialize detector.
        
        Args:
            model_path: Path to a detection model (optional, not used in this implementation)
        """
        # This implementation uses OpenCV for detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info("Using OpenCV for math region detection")
        
        if model_path:
            logger.info(f"Note: Model path {model_path} provided but not used in this implementation")
    
    def detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect math regions in a document image.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            List of detected regions, each a dict with 'bbox' and 'confidence' keys
        """
        height, width = image.shape[:2]
        
        try:
            import cv2
            return self._opencv_detect_regions(image)
        except ImportError:
            # Basic fallback if OpenCV is not available
            logger.warning("OpenCV not available, using basic fallback")
            regions = [
                {
                    'bbox': [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
                    'confidence': 0.5,  # Low confidence for fallback
                    'type': 'math'
                }
            ]
            return regions
    
    def _opencv_detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """OpenCV-based implementation for region detection.
        
        Args:
            image: Input image
            
        Returns:
            Detected regions
        """
        import cv2
        
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Invert image if it's white on black
        if np.mean(gray) < 127:
            gray = 255 - gray
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphology to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter based on size and aspect ratio
            min_area = 1000
            min_aspect_ratio = 0.2
            max_aspect_ratio = 5.0
            
            if (area > min_area and 
                aspect_ratio > min_aspect_ratio and 
                aspect_ratio < max_aspect_ratio):
                
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                regions.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.8,  # Medium confidence for OpenCV detection
                    'type': 'math'
                })
        
        # If no regions found, use the whole image
        if not regions:
            logger.info("No regions detected, using entire image")
            regions = [{
                'bbox': [0, 0, width, height],
                'confidence': 0.5,
                'type': 'math'
            }]
            
        return regions
    
    def extract_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract region images from the document image.
        
        Args:
            image: Input image as NumPy array
            regions: List of detected regions
            
        Returns:
            List of region images
        """
        region_images = []
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Ensure coordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Skip invalid regions
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid region coordinates: {x1}, {y1}, {x2}, {y2}")
                continue
                
            region_img = image[y1:y2, x1:x2]
            region_images.append(region_img)
        
        return region_images