#!/usr/bin/env python3
"""
Script to analyze image sizes in MathWriting and CROHME datasets.

This script:
1. Recursively scans the specified dataset directories
2. Opens each image file and extracts its dimensions
3. Aggregates and prints statistics about image sizes

Example usage:
  python analyze_dataset_sizes.py --mathwriting-dir /path/to/MathWriting --crohme-dir /path/to/CROHME

For Hugging Face datasets, use the --from-huggingface flag:
  python analyze_dataset_sizes.py --from-huggingface
"""

import os
import argparse
from collections import Counter
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from PIL import Image, UnidentifiedImageError
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define image file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    return os.path.splitext(filename)[1] in IMAGE_EXTENSIONS


def get_image_size(file_path: str) -> Optional[Tuple[int, int]]:
    """
    Get the dimensions (width, height) of an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (width, height) or None if the file couldn't be opened as an image
    """
    try:
        with Image.open(file_path) as img:
            return img.size  # (width, height)
    except (UnidentifiedImageError, IOError, OSError) as e:
        logger.warning(f"Could not open {file_path}: {e}")
        return None


def analyze_directory(directory: str, max_files: Optional[int] = None) -> Dict[Tuple[int, int], int]:
    """
    Recursively scan a directory for images and analyze their sizes.
    
    Args:
        directory: Root directory to scan
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        Dictionary mapping image sizes (width, height) to their frequencies
    """
    logger.info(f"Analyzing images in {directory}")
    size_counter = Counter()
    file_count = 0
    
    # Get list of all image files recursively
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
    
    # Process files with progress bar
    total_files = min(len(image_files), max_files) if max_files else len(image_files)
    logger.info(f"Found {total_files} image files to process")
    
    for file_path in tqdm(image_files[:max_files] if max_files else image_files, desc="Processing images"):
        size = get_image_size(file_path)
        if size:
            size_counter[size] += 1
            file_count += 1
    
    logger.info(f"Successfully processed {file_count} images")
    return dict(size_counter)


def analyze_huggingface_dataset(dataset_name: str, sample_size: int = 1000) -> Dict[Tuple[int, int], int]:
    """
    Analyze image sizes from a Hugging Face dataset.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        sample_size: Number of samples to analyze from each split
        
    Returns:
        Dictionary mapping image sizes (width, height) to their frequencies
    """
    from datasets import load_dataset
    
    logger.info(f"Loading {dataset_name} from Hugging Face...")
    dataset = load_dataset(dataset_name)
    
    size_counter = Counter()
    for split_name in dataset.keys():
        logger.info(f"Analyzing '{split_name}' split")
        
        # Determine sample size (all examples or limited sample)
        num_examples = min(len(dataset[split_name]), sample_size)
        
        # Get random samples from the dataset
        indices = list(range(min(len(dataset[split_name]), num_examples)))
        samples = dataset[split_name].select(indices)
        
        # Process images with progress bar
        for example in tqdm(samples, desc=f"Processing {split_name}"):
            if 'image' in example:
                img = example['image']
                size = (img.width, img.height)
                size_counter[size] += 1
    
    return dict(size_counter)


def print_size_summary(sizes: Dict[Tuple[int, int], int], dataset_name: str):
    """
    Print summary of image sizes and their frequencies.
    
    Args:
        sizes: Dictionary mapping sizes to their frequencies
        dataset_name: Name of the dataset for display
    """
    if not sizes:
        logger.warning(f"No valid images found in {dataset_name}")
        return
    
    # Sort sizes by frequency (most common first)
    sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total images
    total_images = sum(sizes.values())
    
    # Print header
    print(f"\n{'=' * 60}")
    print(f"Image Size Summary for {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Total images: {total_images}")
    print(f"Unique sizes: {len(sizes)}")
    print("\nSize Distribution (width × height):")
    print(f"{'Size':^20} | {'Count':^8} | {'Percentage':^12}")
    print(f"{'-' * 20} | {'-' * 8} | {'-' * 12}")
    
    # Print each size and its frequency
    for (width, height), count in sorted_sizes:
        percentage = (count / total_images) * 100
        print(f"{width:4d} × {height:<13d} | {count:8d} | {percentage:9.2f}%")
    
    # Identify min and max dimensions
    min_width = min(width for (width, _), _ in sorted_sizes)
    max_width = max(width for (width, _), _ in sorted_sizes)
    min_height = min(height for (_, height), _ in sorted_sizes)
    max_height = max(height for (_, height), _ in sorted_sizes)
    
    # Print dimension range
    print(f"\nDimension ranges:")
    print(f"Width:  {min_width} to {max_width} pixels")
    print(f"Height: {min_height} to {max_height} pixels")
    
    # Print most common sizes
    print("\nMost common sizes:")
    for i, ((width, height), count) in enumerate(sorted_sizes[:5], 1):
        percentage = (count / total_images) * 100
        print(f"{i}. {width} × {height}: {count} images ({percentage:.2f}%)")
    
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze image sizes in datasets")
    parser.add_argument("--mathwriting-dir", type=str, help="Path to MathWriting dataset directory")
    parser.add_argument("--crohme-dir", type=str, help="Path to CROHME dataset directory")
    parser.add_argument("--max-files", type=int, help="Maximum files to process per dataset (for testing)")
    parser.add_argument("--from-huggingface", action="store_true", help="Load datasets from Hugging Face")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size for Hugging Face datasets")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.from_huggingface and not (args.mathwriting_dir or args.crohme_dir):
        parser.error("Please specify at least one dataset directory or use --from-huggingface")
    
    start_time = time.time()
    
    # Process datasets based on source
    if args.from_huggingface:
        # Process Hugging Face datasets
        logger.info("Analyzing datasets from Hugging Face")
        
        # MathWriting dataset
        try:
            mathwriting_sizes = analyze_huggingface_dataset(
                'andito/mathwriting-google', 
                sample_size=args.sample_size
            )
            print_size_summary(mathwriting_sizes, "MathWriting (Hugging Face)")
        except Exception as e:
            logger.error(f"Error analyzing MathWriting from Hugging Face: {e}")
        
        # CROHME dataset
        try:
            crohme_sizes = analyze_huggingface_dataset(
                'Neeze/CROHME-full', 
                sample_size=args.sample_size
            )
            print_size_summary(crohme_sizes, "CROHME (Hugging Face)")
        except Exception as e:
            logger.error(f"Error analyzing CROHME from Hugging Face: {e}")
    else:
        # Process local directories
        # MathWriting dataset
        if args.mathwriting_dir:
            mathwriting_sizes = analyze_directory(args.mathwriting_dir, args.max_files)
            print_size_summary(mathwriting_sizes, "MathWriting")
        
        # CROHME dataset
        if args.crohme_dir:
            crohme_sizes = analyze_directory(args.crohme_dir, args.max_files)
            print_size_summary(crohme_sizes, "CROHME")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()