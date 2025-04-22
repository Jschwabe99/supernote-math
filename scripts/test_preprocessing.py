#!/usr/bin/env python3
"""
Script to test image preprocessing function with sample images from HuggingFace.

This script:
1. Loads sample images directly from HuggingFace datasets (CROHME and MathWriting)
2. Applies preprocessing to each image with configurable parameters
3. Displays/saves the original and processed images side by side for visual comparison
4. Measures preprocessing performance (time per image, memory usage)
5. Supports bulk processing of large datasets with parallel execution
6. Can save preprocessed images to disk as NumPy arrays for training

Key Features:
- Configurable resolution (default: 512x1024)
- Dataset-specific handling (CROHME vs MathWriting)
- Parallel processing with ThreadPoolExecutor
- Memory-efficient chunked processing
- Performance benchmarking and reporting
- Disk-based persistent storage of preprocessed images
- Command-line interface with extensive configuration options

Usage Examples:
  # Test with small sample set and visualize results
  python test_preprocessing.py --height 512 --width 1024 --output-dir ./assets/test_output

  # Process larger dataset and save to disk
  python test_preprocessing.py --use-huggingface --test-bulk --bulk-samples 1000 \\
    --height 512 --width 1024 --num-workers 8 --save-processed

  # Process wide formulas to test aspect ratio handling
  python test_preprocessing.py --find-wide-formulas --num-samples 20
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import cv2
import random
import time
import psutil
import hashlib
from pathlib import Path
from datasets import load_dataset
import concurrent.futures

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.data.loaders import preprocess_image, preprocess_image_batch
from config import CROHME_DATA_PATH, MATHWRITING_DATA_PATH

# Get directory for storing preprocessed images
PREPROCESSED_DIR = os.environ.get(
    "PREPROCESSED_DIR", 
    str(Path(__file__).resolve().parent.parent / "preprocessed_data")
)
# Ensure directory exists
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
print(f"Preprocessed images will be saved to: {PREPROCESSED_DIR}")

def load_test_images_from_huggingface(num_samples=10, use_local=False, 
                                     crohme_path=None, mathwriting_path=None,
                                     find_wide_formulas=False):
    """
    Load sample images from HuggingFace datasets.
    
    Args:
        num_samples: Number of samples to load from each dataset
        use_local: Whether to try loading from local paths first
        crohme_path: Path to local CROHME dataset (if use_local=True)
        mathwriting_path: Path to local MathWriting dataset (if use_local=True)
        find_wide_formulas: If True, actively seek out wide formulas for testing
        
    Returns:
        Dictionary mapping image names to loaded images
    """
    images = {}
    
    # Try loading from local paths first if requested
    if use_local:
        local_images = load_test_images_from_local(crohme_path, mathwriting_path, num_samples)
        if local_images:
            print(f"Loaded {len(local_images)} images from local paths")
            return local_images
    
    # Load from HuggingFace if local loading failed or wasn't requested
    print(f"Loading {num_samples} samples from each HuggingFace dataset...")
    
    # Load CROHME samples
    try:
        print("Loading CROHME dataset from HuggingFace...")
        crohme_dataset = load_dataset("Neeze/CROHME-full", split="train")
        
        # Get random indices
        indices = random.sample(range(len(crohme_dataset)), min(num_samples, len(crohme_dataset)))
        samples = crohme_dataset.select(indices)
        
        # Extract images
        for idx, sample in enumerate(samples):
            if 'image' in sample:
                # Convert PIL image to numpy array
                img_array = np.array(sample['image'])
                images[f"CROHME_hf_{idx}"] = img_array
                if find_wide_formulas:
                    print(f"CROHME_hf_{idx} dimensions: {img_array.shape}")
        
        print(f"Loaded {len(samples)} CROHME samples")
    except Exception as e:
        print(f"Error loading CROHME from HuggingFace: {e}")
    
    # Load MathWriting samples
    try:
        print("Loading MathWriting dataset from HuggingFace...")
        math_dataset = load_dataset("andito/mathwriting-google", split="train")
        
        if find_wide_formulas:
            # Look specifically for very wide formulas
            print("Searching for wide formulas in MathWriting dataset...")
            wide_indices = []
            # Sample more to find wide formulas
            search_size = min(200, len(math_dataset))
            check_indices = random.sample(range(len(math_dataset)), search_size)
            
            for idx in check_indices:
                sample = math_dataset[idx]
                if 'image' in sample:
                    img = np.array(sample['image'])
                    # Consider a formula wide if width is at least 3x height
                    if len(img.shape) >= 2 and img.shape[1] > img.shape[0] * 3:
                        wide_indices.append(idx)
                        print(f"Found wide formula at index {idx}: {img.shape}")
                        if len(wide_indices) >= num_samples:
                            break
            
            if wide_indices:
                print(f"Using {len(wide_indices)} wide formulas")
                samples = math_dataset.select(wide_indices)
            else:
                print("No very wide formulas found in sample, using random selection")
                indices = random.sample(range(len(math_dataset)), min(num_samples, len(math_dataset)))
                samples = math_dataset.select(indices)
        else:
            # Use random sampling
            indices = random.sample(range(len(math_dataset)), min(num_samples, len(math_dataset)))
            samples = math_dataset.select(indices)
        
        # Extract images
        for idx, sample in enumerate(samples):
            if 'image' in sample:
                # Convert PIL image to numpy array
                img_array = np.array(sample['image'])
                images[f"MathWriting_hf_{idx}"] = img_array
                if find_wide_formulas:
                    print(f"Selected MathWriting_hf_{idx}: {img_array.shape}")
        
        print(f"Loaded {len(samples)} MathWriting samples")
    except Exception as e:
        print(f"Error loading MathWriting from HuggingFace: {e}")
    
    return images

def load_test_images_from_local(crohme_path, mathwriting_path, max_images=3):
    """
    Load sample images from local dataset directories.
    
    Args:
        crohme_path: Path to CROHME dataset
        mathwriting_path: Path to MathWriting dataset
        max_images: Maximum number of images to load from each dataset
    
    Returns:
        Dictionary mapping image paths to loaded images
    """
    images = {}
    
    # Load CROHME images
    if crohme_path and os.path.exists(crohme_path):
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(crohme_path, split, 'images')
            if os.path.exists(split_path):
                for i, img_file in enumerate(os.listdir(split_path)):
                    if i >= max_images:
                        break
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(split_path, img_file)
                        try:
                            img = np.array(Image.open(img_path))
                            images[f"CROHME_{split}_{img_file}"] = img
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                break  # Only process first available split
    
    # Load MathWriting images
    if mathwriting_path and os.path.exists(mathwriting_path):
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(mathwriting_path, split, 'images')
            if os.path.exists(split_path):
                for i, img_file in enumerate(os.listdir(split_path)):
                    if i >= max_images:
                        break
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(split_path, img_file)
                        try:
                            img = np.array(Image.open(img_path))
                            images[f"MathWriting_{split}_{img_file}"] = img
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                break  # Only process first available split
    
    return images

def process_images_parallel(images, resolution=(512, 1024), threshold=None, 
                        crop_margin=0.05, min_margin_pixels=20, num_workers=None,
                        save_processed=False):
    """
    Process images in parallel for better performance
    
    Args:
        images: Dictionary mapping image names to image arrays
        resolution: Tuple of (height, width) for output image
        threshold: Custom threshold value (0-255) or None for defaults
        crop_margin: Margin to add around content as fraction of max dimension
        min_margin_pixels: Minimum margin in pixels
        num_workers: Number of workers for parallel processing
        save_processed: Whether to save processed images to disk
        
    Returns:
        Dictionary mapping image names to processed images
    """
    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)
    
    start_time = time.time()
    results = {}
        
    def process_single_image(name_img_pair):
        """Process a single image and return the name and processed image"""
        name, img = name_img_pair
        try:
            # Determine if this is CROHME (needs inversion)
            needs_inversion = 'crohme' in name.lower()
            
            # Process the image
            processed = preprocess_image(
                image=img, 
                target_resolution=resolution,
                invert=needs_inversion,
                normalize=False,  # Keep as 0-255 for visualization
                threshold=threshold,
                crop_margin=crop_margin,
                min_margin_pixels=min_margin_pixels,
                use_cache=False  # Don't cache internally
            )
            
            # Save processed image if requested
            if save_processed and processed is not None:
                # Generate unique filename from the image content and parameters
                img_hash = hashlib.md5(name.encode()).hexdigest()
                dataset_prefix = "crohme" if needs_inversion else "mathwriting"
                filename = f"{dataset_prefix}_{img_hash}_{resolution[0]}x{resolution[1]}.npy"
                filepath = os.path.join(PREPROCESSED_DIR, filename)
                
                # Save as numpy array
                try:
                    np.save(filepath, processed)
                    print(f"Saved processed image to {filepath}")
                except Exception as save_error:
                    print(f"Error saving processed image {name}: {save_error}")
            
            return name, processed
        except Exception as e:
            print(f"Error processing {name}: {e}")
            return name, None
    
    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for name, processed in executor.map(process_single_image, images.items()):
            if processed is not None:
                results[name] = processed
    
    elapsed_time = time.time() - start_time
    
    print(f"Processed {len(results)} images in {elapsed_time:.2f}s ({elapsed_time/len(images):.4f}s per image)")
    return results

def test_preprocessing(images, resolution=(512, 1024), threshold=None, 
                    crop_margin=0.05, min_margin_pixels=20, output_dir=None,
                    save_only=None, use_parallel=True, num_workers=None,
                    save_processed=False):
    """
    Test preprocessing on sample images with configurable parameters.
    
    Args:
        images: Dictionary mapping image names to image arrays
        resolution: Tuple of (height, width) for output image
        threshold: Custom threshold value (0-255) or None for defaults
        crop_margin: Margin to add around content as fraction of max dimension
        min_margin_pixels: Minimum margin in pixels
        output_dir: Directory to save output images (if None, display instead)
        save_only: If set, save only the first N images to disk (process all for timing)
        use_parallel: Whether to use parallel processing
        num_workers: Number of workers for parallel processing
        save_processed: Whether to save preprocessed images to disk as NumPy arrays
    """
    if not images:
        print("No images to process")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    print(f"Processing {len(images)} images...")
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process images
    if use_parallel:
        # Process all images in parallel
        processed_images = process_images_parallel(
            images, resolution, threshold, crop_margin, min_margin_pixels, num_workers,
            save_processed=save_processed
        )
    else:
        # Process images sequentially
        processed_images = {}
        processing_times = []
        
        for idx, (name, img) in enumerate(images.items()):
            # Determine if this is CROHME (needs inversion)
            needs_inversion = 'crohme' in name.lower()
            
            # Measure processing time
            start_proc_time = time.time()
            
            # Process the image with configurable parameters
            processed = preprocess_image(
                image=img, 
                target_resolution=resolution,
                invert=needs_inversion,
                normalize=False,  # Keep as 0-255 for visualization
                threshold=threshold,
                crop_margin=crop_margin,
                min_margin_pixels=min_margin_pixels,
                use_cache=False  # Don't cache for testing
            )
            
            # Record processing time
            proc_time = time.time() - start_proc_time
            processing_times.append(proc_time)
            processed_images[name] = processed
            
            # Print progress for sequential processing
            if idx % 10 == 0 and idx > 0:
                print(f"Processed {idx}/{len(images)} images...")
    
    # Record final processing time and memory usage
    elapsed_time = time.time() - start_time
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    
    print(f"Total processing time: {elapsed_time:.2f}s for {len(images)} images "
         f"({elapsed_time/len(images):.4f}s per image)")
    print(f"Memory usage: {memory_used:.2f} MB")
    
    # Save/display images
    for idx, (name, processed) in enumerate(processed_images.items()):
        # Skip visualization if we're not saving to disk or we've reached save_only limit
        should_save = output_dir and (save_only is None or idx < save_only)
        if not should_save:
            continue
        
        # Original image
        img = images[name]
        
        # Convert to uint8 for display/saving if needed
        if processed.dtype != np.uint8:
            processed = (processed * 255).astype(np.uint8)
        
        # Create a wider figure for the new dimensions
        # Adjust the figsize based on the aspect ratio
        fig_width = 20  # Wider figure for 512x2048 images
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
        
        # Display original image (grayscale or RGB)
        if len(img.shape) == 2:
            axes[0].imshow(img, cmap='gray')
        elif img.shape[2] == 1:
            axes[0].imshow(img[:,:,0], cmap='gray')
        else:
            axes[0].imshow(img)
        
        axes[0].set_title(f"Original: {img.shape}")
        axes[0].axis('off')
        
        # Display processed image
        processed_display = processed.squeeze()
        
        # Binary display (0=ink, 255=paper)
        axes[1].imshow(processed_display, cmap='gray', vmin=0, vmax=255)
        
        # Show processed dimensions
        shape_info = f"Shape: {processed.shape[0]}×{processed.shape[1]}" if len(processed.shape) == 2 else f"Shape: {processed.shape[0]}×{processed.shape[1]}"
        axes[1].set_title(f"Processed: {shape_info}\nBinary output: 0=ink, 255=paper")
        axes[1].axis('off')
        
        plt.suptitle(f"{name} (invert={'crohme' in name.lower()})")
        plt.tight_layout()
        
        if output_dir:
            out_path = os.path.join(output_dir, f"{name.replace('/', '_')}_comparison.png")
            fig.savefig(out_path, dpi=150)
            plt.close()
        else:
            plt.show()

def test_bulk_processing(num_samples=100, resolution=(512, 1024), use_cache=True, 
                       num_workers=None, chunk_size=20, save_processed=False):
    """
    Test bulk processing performance on a larger set of images.
    
    This function:
    1. Loads samples from CROHME and MathWriting datasets from HuggingFace
    2. Processes them in memory-efficient chunks to avoid RAM exhaustion
    3. Performs dataset-specific preprocessing (inversion for CROHME, enhancement for MathWriting)
    4. Optionally saves the preprocessed images to disk as NumPy arrays
    5. Reports detailed performance metrics throughout processing
    
    The chunked processing approach allows handling very large datasets by processing
    and saving smaller batches sequentially, which is critical for the full ~500K image dataset.
    
    Args:
        num_samples: Number of samples to process from each dataset
        resolution: Target resolution (height, width) for preprocessed images
        use_cache: Whether to use disk caching for faster repeated runs
        num_workers: Number of parallel workers (defaults to min(CPU count, 8))
        chunk_size: Size of chunks to process at once (memory efficiency tradeoff)
        save_processed: Whether to save processed images to PREPROCESSED_DIR as NumPy arrays
    
    Returns:
        None, but prints detailed performance statistics
    
    Performance Notes:
        - Optimal chunk_size depends on available RAM and image resolution
        - For 512x1024 images, chunk_size=20 uses ~1GB RAM with 8 workers
        - Processing time: ~0.0027s per image with parallelization
        - Storage: ~512KB per image, ~250GB for the full dataset (~500K images)
    """
    print(f"Testing bulk processing of {num_samples} samples with {num_workers or 'auto'} workers...")
    print(f"Resolution: {resolution[0]}x{resolution[1]}, Cache: {use_cache}, Chunk size: {chunk_size}")
    
    # Load datasets
    try:
        print("Loading datasets...")
        start_time = time.time()
        
        # Load CROHME samples
        crohme_dataset = load_dataset("Neeze/CROHME-full", split="train")
        crohme_indices = random.sample(range(len(crohme_dataset)), min(num_samples, len(crohme_dataset)))
        crohme_samples = crohme_dataset.select(crohme_indices)
        
        # Load MathWriting samples
        math_dataset = load_dataset("andito/mathwriting-google", split="train")
        math_indices = random.sample(range(len(math_dataset)), min(num_samples, len(math_dataset)))
        math_samples = math_dataset.select(math_indices)
        
        print(f"Loaded datasets in {time.time() - start_time:.2f}s")
        
        # Process CROHME samples in chunks
        print("\nProcessing CROHME samples...")
        crohme_images = []
        for i in range(0, len(crohme_samples), chunk_size):
            chunk = crohme_samples.select(range(i, min(i + chunk_size, len(crohme_samples))))
            chunk_images = [np.array(sample['image']) for sample in chunk]
            chunk_names = [f"CROHME_bulk_{i+idx}" for idx in range(len(chunk))]
            
            start_chunk = time.time()
            
            # Process in different ways based on whether we want to save
            if save_processed:
                # Process and save each image
                processed_count = 0
                for j, (img, name) in enumerate(zip(chunk_images, chunk_names)):
                    try:
                        # Process with inversion for CROHME
                        processed = preprocess_image(
                            image=img,
                            target_resolution=resolution,
                            invert=True,  # CROHME needs inversion
                            normalize=False
                        )
                        
                        if processed is not None:
                            # Save to disk
                            img_hash = hashlib.md5(name.encode()).hexdigest()
                            filename = f"crohme_{img_hash}_{resolution[0]}x{resolution[1]}.npy"
                            filepath = os.path.join(PREPROCESSED_DIR, filename)
                            np.save(filepath, processed)
                            processed_count += 1
                            if (j+1) % 10 == 0 or j == len(chunk_images)-1:
                                print(f"Saved {j+1}/{len(chunk_images)} CROHME images in chunk {i//chunk_size + 1}")
                    except Exception as e:
                        print(f"Error processing {name}: {e}")
                
                # Track progress
                print(f"Chunk {i//chunk_size + 1}/{(len(crohme_samples)-1)//chunk_size + 1}: "
                     f"Processed and saved {processed_count}/{len(chunk_images)} images in {time.time() - start_chunk:.2f}s")
            else:
                # Just process without saving
                results = preprocess_image_batch(
                    chunk_images, 
                    target_resolution=resolution,
                    num_workers=num_workers, 
                    use_cache=use_cache
                )
                
                crohme_images.extend(results)
                print(f"Chunk {i//chunk_size + 1}/{(len(crohme_samples)-1)//chunk_size + 1}: "
                     f"Processed {len(chunk_images)} images in {time.time() - start_chunk:.2f}s")
        
        # Process MathWriting samples in chunks
        print("\nProcessing MathWriting samples...")
        math_images = []
        for i in range(0, len(math_samples), chunk_size):
            chunk = math_samples.select(range(i, min(i + chunk_size, len(math_samples))))
            chunk_images = [np.array(sample['image']) for sample in chunk]
            chunk_names = [f"MathWriting_bulk_{i+idx}" for idx in range(len(chunk))]
            
            start_chunk = time.time()
            
            # Process in different ways based on whether we want to save
            if save_processed:
                # Process and save each image
                processed_count = 0
                for j, (img, name) in enumerate(zip(chunk_images, chunk_names)):
                    try:
                        # Process without inversion for MathWriting
                        processed = preprocess_image(
                            image=img,
                            target_resolution=resolution,
                            invert=False,  # MathWriting doesn't need inversion
                            normalize=False
                        )
                        
                        if processed is not None:
                            # Save to disk
                            img_hash = hashlib.md5(name.encode()).hexdigest()
                            filename = f"mathwriting_{img_hash}_{resolution[0]}x{resolution[1]}.npy"
                            filepath = os.path.join(PREPROCESSED_DIR, filename)
                            np.save(filepath, processed)
                            processed_count += 1
                            if (j+1) % 10 == 0 or j == len(chunk_images)-1:
                                print(f"Saved {j+1}/{len(chunk_images)} MathWriting images in chunk {i//chunk_size + 1}")
                    except Exception as e:
                        print(f"Error processing {name}: {e}")
                
                # Track progress
                print(f"Chunk {i//chunk_size + 1}/{(len(math_samples)-1)//chunk_size + 1}: "
                     f"Processed and saved {processed_count}/{len(chunk_images)} images in {time.time() - start_chunk:.2f}s")
            else:
                # Just process without saving
                results = preprocess_image_batch(
                    chunk_images, 
                    target_resolution=resolution,
                    num_workers=num_workers, 
                    use_cache=use_cache
                )
                
                math_images.extend(results)
                print(f"Chunk {i//chunk_size + 1}/{(len(math_samples)-1)//chunk_size + 1}: "
                     f"Processed {len(chunk_images)} images in {time.time() - start_chunk:.2f}s")
            
        # Report statistics
        total_images = len(crohme_images) + len(math_images)
        print(f"\nProcessed {total_images} images in total")
        
    except Exception as e:
        print(f"Error in bulk processing test: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test image preprocessing function")
    parser.add_argument("--crohme-dir", type=str, help="Path to local CROHME dataset directory")
    parser.add_argument("--mathwriting-dir", type=str, help="Path to local MathWriting dataset directory")
    parser.add_argument("--height", type=int, default=512, help="Target height for preprocessing")
    parser.add_argument("--width", type=int, default=1024, help="Target width for preprocessing")
    parser.add_argument("--threshold", type=int, help="Custom threshold value (0-255)")
    parser.add_argument("--crop-margin", type=float, default=0.05, 
                       help="Margin around content as fraction of max dimension")
    parser.add_argument("--min-margin", type=int, default=20, 
                       help="Minimum margin in pixels")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to load from each dataset")
    parser.add_argument("--output-dir", type=str, default="./assets/test_output", help="Directory to save output images")
    parser.add_argument("--use-local", action="store_true", help="Try to use local datasets first")
    parser.add_argument("--use-huggingface", action="store_true", default=True, help="Use HuggingFace datasets")
    parser.add_argument("--find-wide-formulas", action="store_true", help="Actively search for wide formulas")
    parser.add_argument("--save-only", type=int, default=None, 
                        help="Save only the first N samples, process all but limit disk space")
    parser.add_argument("--timing-only", action="store_true", 
                        help="Only test processing time, don't save images")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="Use parallel processing")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of workers for parallel processing (default: CPU count)")
    parser.add_argument("--test-bulk", action="store_true",
                        help="Test bulk processing on a larger dataset")
    parser.add_argument("--bulk-samples", type=int, default=100,
                        help="Number of samples for bulk processing test")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use disk caching for processed images")
    parser.add_argument("--save-processed", action="store_true",
                        help="Save preprocessed images as NumPy arrays to disk")
    parser.add_argument("--output-dir-processed", type=str,
                        help="Directory to save preprocessed NumPy arrays (default: ./preprocessed_data)")
    
    args = parser.parse_args()
    
    # Create output directories
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Set preprocessed output directory if specified
    if args.output_dir_processed:
        global PREPROCESSED_DIR
        PREPROCESSED_DIR = args.output_dir_processed
        os.makedirs(PREPROCESSED_DIR, exist_ok=True)
        print(f"Preprocessed images will be saved to: {PREPROCESSED_DIR}")
    
    # If paths are not provided, try to use default paths
    crohme_path = args.crohme_dir or CROHME_DATA_PATH
    mathwriting_path = args.mathwriting_dir or MATHWRITING_DATA_PATH
    
    # Create resolution tuple for preprocessing
    resolution = (args.height, args.width)
    
    if args.test_bulk:
        # Run bulk processing test
        test_bulk_processing(
            num_samples=args.bulk_samples,
            resolution=resolution,
            use_cache=args.use_cache,
            num_workers=args.num_workers,
            save_processed=args.save_processed
        )
    else:
        start_time = time.time()
        
        # Load test images
        if args.use_huggingface:
            print("Loading test images from HuggingFace datasets...")
            images = load_test_images_from_huggingface(
                num_samples=args.num_samples,
                use_local=args.use_local,
                crohme_path=crohme_path,
                mathwriting_path=mathwriting_path,
                find_wide_formulas=args.find_wide_formulas
            )
        else:
            print(f"Loading test images from local paths: CROHME ({crohme_path}) and MathWriting ({mathwriting_path})...")
            images = load_test_images_from_local(crohme_path, mathwriting_path, args.num_samples)
        
        if not images:
            print("No images found. Check dataset paths or HuggingFace connection.")
            return
        
        print(f"Testing preprocessing on {len(images)} images with:")
        print(f"  - Resolution: {args.height}×{args.width}")
        print(f"  - Threshold: {args.threshold if args.threshold is not None else 'default'}")
        print(f"  - Crop margin: {args.crop_margin:.1%} (min {args.min_margin}px)")
        print(f"  - Parallel processing: {'On' if args.parallel else 'Off'}")
        print(f"  - Workers: {args.num_workers if args.num_workers else 'auto'}")
        
        # Determine output directory based on timing_only flag
        output_dir = None if args.timing_only else args.output_dir
        
        # Test preprocessing with all parameters
        test_preprocessing(
            images=images, 
            resolution=resolution,
            threshold=args.threshold,
            crop_margin=args.crop_margin,
            min_margin_pixels=args.min_margin,
            output_dir=output_dir,
            save_only=args.save_only,
            use_parallel=args.parallel,
            num_workers=args.num_workers,
            save_processed=args.save_processed
        )
        
        elapsed_time = time.time() - start_time
        print(f"Done! Overall process completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()