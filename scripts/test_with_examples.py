#!/usr/bin/env python3
"""
Test script for the PosFormer-based math recognition pipeline using real examples.

This script:
1. Loads test images with handwritten mathematical expressions
2. Processes individual expressions or segments a page into multiple expressions
3. Runs each expression through the PosFormer model
4. Displays the recognized LaTeX output and solutions
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import json
import time
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import MathRecognitionPipeline, PosFormerRecognizer

def segment_expressions(image, min_area=1000, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """
    Segment a page into individual math expressions using contour detection.
    
    Args:
        image: Input image array (grayscale)
        min_area: Minimum area of an expression region
        min_aspect_ratio: Minimum width/height ratio to be considered valid
        max_aspect_ratio: Maximum width/height ratio to be considered valid
        
    Returns:
        List of (x, y, w, h) bounding boxes for detected expressions
    """
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
    
    # Apply some morphology to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    expression_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        if (area > min_area and 
            aspect_ratio > min_aspect_ratio and 
            aspect_ratio < max_aspect_ratio):
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            
            expression_boxes.append((x, y, w, h))
    
    # Sort boxes from top to bottom
    expression_boxes.sort(key=lambda box: box[1])
    
    return expression_boxes

def process_page(image_path, model_path=None, show_result=True):
    """
    Process a page with multiple math expressions.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model checkpoint
        show_result: Whether to display the results
    """
    print(f"Processing page with math expressions: {image_path}")
    
    try:
        # Load the image
        image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
        
        # Segment expressions
        expression_boxes = segment_expressions(image)
        print(f"Detected {len(expression_boxes)} expressions")
        
        # Initialize the pipeline
        pipeline = MathRecognitionPipeline(posformer_model_path=model_path)
        
        # Process each expression
        results = []
        for i, (x, y, w, h) in enumerate(expression_boxes):
            # Extract expression region
            expr_image = image[y:y+h, x:x+w]
            
            # Process the expression
            start_time = time.time()
            result = pipeline.recognizer.recognize(expr_image)
            inference_time = time.time() - start_time
            
            # Add position info to result
            result['position'] = (x, y, w, h)
            result['index'] = i + 1
            
            # Try to solve the expression
            solution = pipeline.solver.solve_latex(result['latex'])
            result['solution'] = solution
            
            results.append(result)
            
            # Print results
            print(f"Expression {i+1}:")
            print(f"  Position: x={x}, y={y}, w={w}, h={h}")
            print(f"  Recognized: {result['latex']}")
            print(f"  Confidence: {result.get('confidence', 0):.4f}")
            print(f"  Inference time: {inference_time:.4f}s")
            
            if solution['solved']:
                print(f"  Solution: {solution['result_latex']}")
            else:
                print(f"  Could not solve: {solution.get('error', 'Unknown error')}")
            print()
        
        # Show the results
        if show_result and results:
            plt.figure(figsize=(12, 10))
            
            # Display original image with bounding boxes
            plt.imshow(image, cmap='gray')
            
            # Add bounding boxes and labels
            for result in results:
                x, y, w, h = result['position']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add label with recognition result
                label = f"{result['index']}: {result['latex']}"
                if result['solution']['solved']:
                    label += f" = {result['solution']['result_latex']}"
                plt.text(x, y-5, label, color='red', fontsize=8, 
                         bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title(f"Math Expressions Recognition")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return results
    
    except Exception as e:
        print(f"Error processing page: {e}")
        return []

def process_single_expression(image_path, model_path=None, show_result=True):
    """
    Process a single math expression image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model checkpoint
        show_result: Whether to display the results
    """
    print(f"Processing single math expression: {image_path}")
    
    try:
        # Load the image
        image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
        
        # Initialize the pipeline
        pipeline = MathRecognitionPipeline(posformer_model_path=model_path)
        
        # Process the expression
        start_time = time.time()
        result = pipeline.recognizer.recognize(image)
        inference_time = time.time() - start_time
        
        # Try to solve the expression
        solution = pipeline.solver.solve_latex(result['latex'])
        
        # Print results
        print(f"Recognized: {result['latex']}")
        print(f"Confidence: {result.get('confidence', 0):.4f}")
        print(f"Inference time: {inference_time:.4f}s")
        
        if solution['solved']:
            print(f"Solution: {solution['result_latex']}")
        else:
            print(f"Could not solve: {solution.get('error', 'Unknown error')}")
        
        # Show the result
        if show_result:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 1, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Input | Recognized: {result['latex']}")
            if solution['solved']:
                plt.xlabel(f"Solution: {solution['result_latex']}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return result
    
    except Exception as e:
        print(f"Error processing expression: {e}")
        return {'latex': '', 'error': str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test PosFormer pipeline with examples')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    parser.add_argument('--page', action='store_true', help='Process as a page with multiple expressions')
    parser.add_argument('--test-page', action='store_true', 
                      help='Use the default Test_Data_Page_1.png')
    args = parser.parse_args()
    
    # Choose image path
    image_path = args.image
    if args.test_page or (not image_path and not args.page):
        image_path = str(Path(__file__).resolve().parent.parent / 'assets' / 'Test_Data_Page_1.png')
        args.page = True
    
    if not image_path:
        print("Error: Please provide an image path with --image or use --test-page")
        return
    
    # Process the image
    if args.page:
        process_page(image_path, args.model, not args.no_show)
    else:
        process_single_expression(image_path, args.model, not args.no_show)

if __name__ == "__main__":
    main()