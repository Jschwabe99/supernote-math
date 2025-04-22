#!/usr/bin/env python3
"""
Script to test the DataLoader with the new preprocessing function.

This script:
1. Sets up the DataLoader with both datasets
2. Loads a small batch of images from each dataset
3. Displays the preprocessed images to verify the processing pipeline
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.data.loaders import DataLoader, preprocess_image
from config import CROHME_DATA_PATH, MATHWRITING_DATA_PATH

def visualize_batch(batch, title):
    """
    Visualize a batch of images from a TensorFlow dataset.
    
    Args:
        batch: Tuple of (images, labels) from a dataset
        title: Title for the figure
    """
    images, labels = batch
    batch_size = len(images)
    
    # Determine grid size based on batch size
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(title, fontsize=16)
    
    for i in range(batch_size):
        plt.subplot(rows, cols, i + 1)
        
        # Get image and convert to displayable format if needed
        img = images[i].numpy()
        
        # Handle normalization - if max value is 1.0, scale to 0-255
        if img.max() <= 1.0:
            img = img * 255
            
        img = img.astype(np.uint8)
        
        # Display the image
        if len(img.shape) == 3 and img.shape[2] == 1:
            plt.imshow(img[:, :, 0], cmap='gray')
        else:
            plt.imshow(img, cmap='gray')
            
        # Display label
        try:
            label = labels[i].numpy().decode('utf-8')
            # Truncate long labels
            if len(label) > 20:
                label = label[:20] + "..."
            plt.title(f"Label: {label}")
        except:
            plt.title("Label: [Error]")
            
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def test_loader(data_dir, batch_size=4, target_size=256):
    """
    Test the DataLoader with both datasets.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for testing
        target_size: Image target size
    """
    print(f"Testing DataLoader with target size ({target_size}, {target_size})")
    
    # Create DataLoader instance
    loader = DataLoader(
        data_dir, 
        batch_size=batch_size,
        input_size=(target_size, target_size),
        use_augmentation=False  # Turn off augmentations for testing
    )
    
    # Load CROHME dataset
    print("Loading CROHME dataset...")
    crohme_dataset = loader.load_crohme_dataset('train')
    
    # Load MathWriting dataset
    print("Loading MathWriting dataset...")
    mathwriting_dataset = loader.load_mathwriting_dataset('train')
    
    # Get a batch from each dataset
    print("Taking sample batches...")
    try:
        crohme_batch = next(iter(crohme_dataset))
        print(f"CROHME batch shape: {crohme_batch[0].shape}")
        visualize_batch(crohme_batch, "CROHME Dataset Sample")
    except Exception as e:
        print(f"Error visualizing CROHME batch: {e}")
    
    try:
        mathwriting_batch = next(iter(mathwriting_dataset))
        print(f"MathWriting batch shape: {mathwriting_batch[0].shape}")
        visualize_batch(mathwriting_batch, "MathWriting Dataset Sample")
    except Exception as e:
        print(f"Error visualizing MathWriting batch: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test DataLoader with preprocessed images")
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--target-size", type=int, default=256, help="Target size for images")
    
    args = parser.parse_args()
    
    # Use provided data directory or default
    data_dir = args.data_dir or Path(os.path.dirname(os.path.abspath(__file__))).parent / "assets"
    
    print(f"Using data directory: {data_dir}")
    test_loader(data_dir, args.batch_size, args.target_size)
    print("Done!")

if __name__ == "__main__":
    main()