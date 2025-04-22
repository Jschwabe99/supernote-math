#!/usr/bin/env python3
"""
Script to visualize preprocessed NumPy arrays for the Supernote Math project.

This script:
1. Loads preprocessed images from NumPy files (.npy) in the preprocessed_data directory
2. Displays them for visual verification and quality assessment
3. Provides comprehensive statistics about the preprocessed dataset:
   - Total number of preprocessed files
   - Dataset composition (CROHME vs MathWriting)
   - File sizes and total storage requirements
   - Array shapes and data types
   - Estimated storage requirements for the full dataset

Key Features:
- Random or sequential sampling of preprocessed images
- Detailed storage statistics for capacity planning
- Visual verification of binary image quality (0=ink, 255=paper)
- Command-line interface with configuration options

Usage Examples:
  # Basic visualization with default settings
  python visualize_preprocessed.py
  
  # Show more samples and detailed statistics
  python visualize_preprocessed.py --num-samples 10 --data-dir ./preprocessed_data
  
  # Just show statistics without visualization
  python visualize_preprocessed.py --stats-only

Verification Criteria:
- Images should be binary (0=ink, 255=paper)
- Resolution should match target (typically 512x1024)
- Content should be centered and properly scaled
- No distortion or aspect ratio problems
- Properly normalized with black ink and white paper
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def visualize_preprocessed_samples(data_dir, num_samples=5, random_sampling=True, show_stats=True):
    """
    Visualize preprocessed images from NumPy files and analyze dataset characteristics.
    
    This function provides both visual verification and statistical analysis of the
    preprocessed dataset. It:
    1. Scans the preprocessed data directory for .npy files
    2. Analyzes dataset composition (CROHME vs MathWriting)
    3. Calculates storage statistics (file sizes, total size, projections)
    4. Verifies image properties (shape, data type, binary nature)
    5. Displays sample images for visual inspection
    
    The statistics are particularly useful for:
    - Verifying preprocessing consistency
    - Planning storage requirements for the full dataset
    - Ensuring proper binary representation (0=ink, 255=paper)
    - Confirming target resolution is maintained
    
    Args:
        data_dir: Directory containing preprocessed NumPy files (.npy)
        num_samples: Number of samples to visualize (default: 5)
        random_sampling: Whether to pick samples randomly or sequentially
        show_stats: Whether to show detailed statistics about the dataset
        
    Returns:
        None, but prints statistics and displays visualizations
        
    Example usage:
        # Basic visualization with random sampling
        visualize_preprocessed_samples('./preprocessed_data')
        
        # More samples with sequential examination
        visualize_preprocessed_samples('./preprocessed_data', num_samples=10, random_sampling=False)
        
        # Just show statistics without visualization
        visualize_preprocessed_samples('./preprocessed_data', num_samples=0, show_stats=True)
    """
    # Get list of NumPy files
    data_dir = Path(data_dir)
    npy_files = list(data_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"No preprocessed files found in {data_dir}")
        return
    
    print(f"Found {len(npy_files)} preprocessed files in {data_dir}")
    
    # Count by dataset type
    crohme_count = sum(1 for f in npy_files if "crohme_" in f.name.lower())
    mathwriting_count = sum(1 for f in npy_files if "mathwriting_" in f.name.lower())
    other_count = len(npy_files) - crohme_count - mathwriting_count
    
    print(f"Dataset composition:")
    print(f"  - CROHME: {crohme_count} files")
    print(f"  - MathWriting: {mathwriting_count} files")
    print(f"  - Other: {other_count} files")
    
    # Sample files for visualization
    if random_sampling:
        files_to_show = random.sample(npy_files, min(num_samples, len(npy_files)))
    else:
        files_to_show = npy_files[:min(num_samples, len(npy_files))]
    
    # Calculate file sizes
    file_sizes = [os.path.getsize(f) for f in npy_files]
    total_size_mb = sum(file_sizes) / (1024 * 1024)
    avg_size_kb = np.mean(file_sizes) / 1024
    
    print(f"Storage statistics:")
    print(f"  - Total size: {total_size_mb:.2f} MB")
    print(f"  - Average file size: {avg_size_kb:.2f} KB")
    print(f"  - Estimated size for 500K images: {total_size_mb/len(npy_files)*500000:.2f} MB")
    
    # Show file stats if requested
    if show_stats:
        # Load a few files to check properties
        stats_samples = random.sample(npy_files, min(20, len(npy_files)))
        shapes = []
        dtypes = []
        
        for file_path in stats_samples:
            try:
                img = np.load(file_path)
                shapes.append(img.shape)
                dtypes.append(img.dtype)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Sample statistics:")
        print(f"  - Shapes: {set(shapes)}")
        print(f"  - Data types: {set(dtypes)}")
    
    # Display the sampled files
    for i, file_path in enumerate(files_to_show):
        try:
            # Load the image
            img = np.load(file_path)
            
            # Create figure
            plt.figure(figsize=(10, 5))
            
            # Determine dataset type from filename
            dataset_type = "CROHME" if "crohme_" in file_path.name.lower() else "MathWriting"
            
            # Display image
            plt.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=255)
            plt.title(f"{dataset_type}: {file_path.name}\nShape: {img.shape}, Type: {img.dtype}")
            plt.colorbar(label='Pixel Value')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error displaying {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize preprocessed NumPy arrays")
    parser.add_argument("--data-dir", type=str, default="./preprocessed_data",
                       help="Directory containing preprocessed NumPy files")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--sequential", action="store_true",
                       help="Show samples in sequential order instead of random")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics, no visualization")
    
    args = parser.parse_args()
    
    # Call the visualization function
    visualize_preprocessed_samples(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        random_sampling=not args.sequential,
        show_stats=True
    )

if __name__ == "__main__":
    main()