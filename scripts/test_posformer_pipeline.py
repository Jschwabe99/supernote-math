#!/usr/bin/env python3
"""
Test script for the PosFormer-based math recognition pipeline.

This script:
1. Creates sample stroke data (simulated handwriting)
2. Converts the strokes to an image
3. Runs the image through the PosFormer model
4. Prints the recognized LaTeX output
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import time
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import MathRecognitionPipeline, Stroke, PosFormerRecognizer

def create_sample_equation(equation_type='quadratic', noise=0.0, complexity=1.0):
    """
    Create sample stroke data for a mathematical equation.
    
    Args:
        equation_type: Type of equation to generate ('linear', 'quadratic', etc.)
        noise: Amount of noise to add to the strokes (0.0 to 1.0)
        complexity: Complexity factor for the equation (1.0 = normal)
        
    Returns:
        List of Stroke objects
    """
    # Base offset for positioning
    base_x = 200
    base_y = 120
    
    # Basic parameters
    stroke_spacing = 40
    x_scale = 1.0
    y_scale = 1.0
    
    # Adjust parameters based on complexity
    stroke_spacing *= complexity
    
    # Function to add some random noise to a point
    def add_noise(x, y, noise_level=noise):
        if noise_level <= 0:
            return x, y
        # Add Gaussian noise
        x_noise = random.gauss(0, noise_level * 5)
        y_noise = random.gauss(0, noise_level * 5)
        return x + x_noise, y + y_noise
    
    strokes = []
    
    if equation_type == 'quadratic':
        # Create a simple quadratic equation: x^2 + 2x + 1 = 0
        
        # Letter "x"
        x_stroke1 = Stroke()
        x_stroke2 = Stroke()
        
        # First diagonal of x
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + (t_normalized * 20 * x_scale)
            y = base_y - (t_normalized * 20 * y_scale)
            x, y = add_noise(x, y)
            x_stroke1.add_point(x, y)
        
        # Second diagonal of x
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + (t_normalized * 20 * x_scale)
            y = base_y + (t_normalized * 20 * y_scale) - 20
            x, y = add_noise(x, y)
            x_stroke2.add_point(x, y)
        
        strokes.append(x_stroke1)
        strokes.append(x_stroke2)
        
        # Superscript 2
        two_stroke = Stroke()
        for t in range(0, 11):
            t_normalized = t / 10.0
            if t_normalized < 0.5:
                # Top curve of 2
                angle = t_normalized * 2 * 3.14159 * 0.25
                x = base_x + 30 * x_scale + 5 * np.cos(angle)
                y = base_y - 25 * y_scale + 5 * np.sin(angle)
            else:
                # Bottom part of 2
                x = base_x + 30 * x_scale - (t_normalized - 0.5) * 2 * 10
                y = base_y - 20 * y_scale + (t_normalized - 0.5) * 2 * 10
            x, y = add_noise(x, y)
            two_stroke.add_point(x, y)
        
        strokes.append(two_stroke)
        
        # Plus sign
        plus_h_stroke = Stroke()
        plus_v_stroke = Stroke()
        
        # Horizontal line of +
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 50 * x_scale + (t_normalized * 20)
            y = base_y
            x, y = add_noise(x, y)
            plus_h_stroke.add_point(x, y)
        
        # Vertical line of +
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 60 * x_scale
            y = base_y - 10 + (t_normalized * 20)
            x, y = add_noise(x, y)
            plus_v_stroke.add_point(x, y)
        
        strokes.append(plus_h_stroke)
        strokes.append(plus_v_stroke)
        
        # Number 2
        two_stroke_main = Stroke()
        for t in range(0, 21):
            t_normalized = t / 20.0
            if t_normalized < 0.3:
                # Top curve of 2
                angle = t_normalized / 0.3 * 3.14159 * 0.5
                x = base_x + 90 * x_scale + 10 * np.cos(angle)
                y = base_y - 10 * y_scale + 10 * np.sin(angle)
            elif t_normalized < 0.6:
                # Middle part of 2
                x = base_x + 100 * x_scale - (t_normalized - 0.3) / 0.3 * 20
                y = base_y
            else:
                # Bottom part of 2
                x = base_x + 80 * x_scale + (t_normalized - 0.6) / 0.4 * 30
                y = base_y + (t_normalized - 0.6) / 0.4 * 15
            x, y = add_noise(x, y)
            two_stroke_main.add_point(x, y)
        
        strokes.append(two_stroke_main)
        
        # Letter "x" (second x)
        x2_stroke1 = Stroke()
        x2_stroke2 = Stroke()
        
        # First diagonal of x
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 120 * x_scale + (t_normalized * 20)
            y = base_y - 10 * y_scale + (t_normalized * 20)
            x, y = add_noise(x, y)
            x2_stroke1.add_point(x, y)
        
        # Second diagonal of x
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 120 * x_scale + (t_normalized * 20)
            y = base_y + 10 * y_scale - (t_normalized * 20)
            x, y = add_noise(x, y)
            x2_stroke2.add_point(x, y)
        
        strokes.append(x2_stroke1)
        strokes.append(x2_stroke2)
        
        # Plus sign (second +)
        plus2_h_stroke = Stroke()
        plus2_v_stroke = Stroke()
        
        # Horizontal line of +
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 160 * x_scale + (t_normalized * 20)
            y = base_y
            x, y = add_noise(x, y)
            plus2_h_stroke.add_point(x, y)
        
        # Vertical line of +
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 170 * x_scale
            y = base_y - 10 + (t_normalized * 20)
            x, y = add_noise(x, y)
            plus2_v_stroke.add_point(x, y)
        
        strokes.append(plus2_h_stroke)
        strokes.append(plus2_v_stroke)
        
        # Number 1
        one_stroke = Stroke()
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 200 * x_scale
            y = base_y + 10 - (t_normalized * 20)
            x, y = add_noise(x, y)
            one_stroke.add_point(x, y)
        
        strokes.append(one_stroke)
        
        # Equals sign
        eq1_stroke = Stroke()
        eq2_stroke = Stroke()
        
        # Top line of =
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 230 * x_scale + (t_normalized * 20)
            y = base_y - 5
            x, y = add_noise(x, y)
            eq1_stroke.add_point(x, y)
        
        # Bottom line of =
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 230 * x_scale + (t_normalized * 20)
            y = base_y + 5
            x, y = add_noise(x, y)
            eq2_stroke.add_point(x, y)
        
        strokes.append(eq1_stroke)
        strokes.append(eq2_stroke)
        
        # Number 0
        zero_stroke = Stroke()
        for t in range(0, 21):
            t_normalized = t / 20.0
            angle = t_normalized * 2 * 3.14159
            x = base_x + 280 * x_scale + 10 * np.cos(angle)
            y = base_y + 10 * np.sin(angle)
            x, y = add_noise(x, y)
            zero_stroke.add_point(x, y)
        
        strokes.append(zero_stroke)
    
    elif equation_type == 'fraction':
        # Create a simple fraction: \frac{1}{2}
        
        # Number 1
        one_stroke = Stroke()
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + 10 * x_scale
            y = base_y - 30 * y_scale + (t_normalized * 20)
            x, y = add_noise(x, y)
            one_stroke.add_point(x, y)
        
        strokes.append(one_stroke)
        
        # Fraction line
        frac_stroke = Stroke()
        for t in range(0, 11):
            t_normalized = t / 10.0
            x = base_x + (t_normalized * 40 * x_scale)
            y = base_y
            x, y = add_noise(x, y)
            frac_stroke.add_point(x, y)
        
        strokes.append(frac_stroke)
        
        # Number 2
        two_stroke = Stroke()
        for t in range(0, 21):
            t_normalized = t / 20.0
            if t_normalized < 0.3:
                # Top curve of 2
                angle = t_normalized / 0.3 * 3.14159 * 0.5
                x = base_x + 10 * x_scale + 10 * np.cos(angle)
                y = base_y + 20 * y_scale + 10 * np.sin(angle)
            elif t_normalized < 0.6:
                # Middle part of 2
                x = base_x + 20 * x_scale - (t_normalized - 0.3) / 0.3 * 20
                y = base_y + 30 * y_scale
            else:
                # Bottom part of 2
                x = base_x + (t_normalized - 0.6) / 0.4 * 30
                y = base_y + 30 * y_scale + (t_normalized - 0.6) / 0.4 * 10
            x, y = add_noise(x, y)
            two_stroke.add_point(x, y)
        
        strokes.append(two_stroke)
        
    return strokes

def test_with_strokes(model_path=None, show_result=True):
    """
    Test the pipeline with generated stroke data.
    
    Args:
        model_path: Path to the model checkpoint
        show_result: Whether to display the results
    """
    print("Testing PosFormer pipeline with simulated strokes...")
    
    # Create sample stroke data
    strokes = create_sample_equation('quadratic', noise=0.2)
    
    # Initialize the pipeline
    pipeline = MathRecognitionPipeline(posformer_model_path=model_path)
    
    # Recognize and solve
    start_time = time.time()
    result = pipeline.process_strokes(strokes)
    total_time = time.time() - start_time
    
    # Print results
    print(f"Recognition time: {result.get('inference_time', 0):.4f}s")
    print(f"Total processing time: {total_time:.4f}s")
    print(f"Recognized LaTeX: {result['latex']}")
    
    if result['solution']['solved']:
        print(f"Solution: {result['solution']['result_latex']}")
    else:
        print(f"Could not solve: {result['solution'].get('error', 'Unknown error')}")
    
    # Show the input image
    if show_result:
        # Render the strokes
        image = pipeline.recognizer.strokes_to_image(strokes)
        
        # Display
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 1, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Input | Recognized: {result['latex']}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def test_with_image(image_path, model_path=None, show_result=True):
    """
    Test the pipeline with an image file.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model checkpoint
        show_result: Whether to display the results
    """
    print(f"Testing PosFormer pipeline with image: {image_path}")
    
    try:
        # Load the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_array = np.array(image)
        
        # Initialize the pipeline
        pipeline = MathRecognitionPipeline(posformer_model_path=model_path)
        
        # Process the image directly
        start_time = time.time()
        
        # Use the recognizer directly for a single image
        result = pipeline.recognizer.recognize(image_array)
        total_time = time.time() - start_time
        
        # Print results
        print(f"Recognition time: {result.get('inference_time', 0):.4f}s")
        print(f"Total processing time: {total_time:.4f}s")
        print(f"Recognized LaTeX: {result['latex']}")
        
        # Try to solve the expression
        solution = pipeline.solver.solve_latex(result['latex'])
        if solution['solved']:
            print(f"Solution: {solution['result_latex']}")
        else:
            print(f"Could not solve: {solution.get('error', 'Unknown error')}")
        
        # Show the input image and result
        if show_result:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 1, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Input | Recognized: {result['latex']}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test PosFormer pipeline')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    args = parser.parse_args()
    
    # Choose between stroke generation and image input
    if args.image:
        test_with_image(args.image, args.model, not args.no_show)
    else:
        test_with_strokes(args.model, not args.no_show)

if __name__ == "__main__":
    main()