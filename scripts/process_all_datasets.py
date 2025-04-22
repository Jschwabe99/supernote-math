#!/usr/bin/env python3
"""
Script to preprocess and cache ALL images in the CROHME and MathWriting datasets.

This script is designed for processing the ENTIRE dataset (~500K images) efficiently:
1. Loads all images from both CROHME and MathWriting datasets
2. Processes them in memory-efficient chunks to prevent OOM errors
3. Uses parallel processing with ThreadPoolExecutor for speed
4. Saves preprocessed images to disk as NumPy arrays (.npy) for fast retrieval
5. Reports detailed progress and timing statistics
6. Handles dataset-specific preprocessing requirements
   - CROHME: Inverts white-on-black to black-on-white
   - MathWriting: Enhances faint strokes and binarizes

Important usage notes:
- This script is designed for COMPLETE dataset processing (not samples)
- Requires ~250GB of storage for the full dataset (~500K images)
- Processing the full dataset will take several hours
- Use the --chunk-size parameter to control memory usage
- Uses the optimized preprocessing function from core/data/loaders.py

Usage examples:
  # Process both datasets with default parameters
  python scripts/process_all_datasets.py

  # Process only CROHME with custom parameters
  python scripts/process_all_datasets.py --dataset crohme --height 512 --width 1024 \\
    --num-workers 8 --chunk-size 500

  # Process only training split of MathWriting
  python scripts/process_all_datasets.py --dataset mathwriting --splits train

Performance metrics:
- Expected processing speed: ~0.0027s per image with 8 workers
- Memory usage: Scales with --chunk-size and --num-workers
- Storage: ~512KB per preprocessed image
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import gc
import logging

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.data.loaders import preprocess_image_batch
from config import CROHME_DATA_PATH, MATHWRITING_DATA_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dataset_processor")

def load_dataset_paths(dataset_dir, dataset_name, split='train'):
    """
    Load image paths for a specific dataset and split.
    
    Args:
        dataset_dir: Path to the dataset directory
        dataset_name: Name of the dataset ('crohme' or 'mathwriting')
        split: Dataset split ('train', 'val', or 'test')
        
    Returns:
        List of image paths
    """
    logger.info(f"Loading {dataset_name} {split} image paths...")
    image_paths = []
    
    dataset_path = Path(dataset_dir) / dataset_name.lower() / split
    
    if not dataset_path.exists():
        logger.warning(f"{dataset_name} {split} path does not exist: {dataset_path}")
        return image_paths
    
    try:
        # Load annotations file
        if dataset_name.lower() == 'crohme':
            anno_file = dataset_path / 'formulas.json'
            images_dir = dataset_path / 'images'
            
            if anno_file.exists():
                with open(anno_file, 'r') as f:
                    annotations = json.load(f)
                
                # Get all image paths
                for filename in tqdm(annotations.keys(), desc=f"Finding {dataset_name} images"):
                    image_path = images_dir / filename
                    if image_path.exists():
                        image_paths.append(str(image_path))
            else:
                logger.warning(f"{dataset_name} annotations file not found: {anno_file}")
                
        elif dataset_name.lower() == 'mathwriting':
            anno_file = dataset_path / 'annotations.json'
            images_dir = dataset_path / 'images'
            
            if anno_file.exists():
                with open(anno_file, 'r') as f:
                    annotations = json.load(f)
                
                # Get all image paths
                for item in tqdm(annotations, desc=f"Finding {dataset_name} images"):
                    image_path = images_dir / item['filename']
                    if image_path.exists():
                        image_paths.append(str(image_path))
            else:
                logger.warning(f"{dataset_name} annotations file not found: {anno_file}")
    
    except Exception as e:
        logger.error(f"Error loading {dataset_name} dataset: {e}")
    
    logger.info(f"Found {len(image_paths)} images in {dataset_name} {split} dataset")
    return image_paths

def process_dataset(image_paths, target_resolution, num_workers, chunk_size, use_cache=True):
    """
    Process a dataset in memory-efficient chunks to handle large datasets.
    
    This function is critical for processing the full dataset without memory issues:
    1. Divides processing into manageable chunks based on chunk_size
    2. Uses parallel processing with ThreadPoolExecutor for each chunk
    3. Tracks detailed performance metrics and progress information
    4. Performs forced garbage collection between chunks to prevent memory leaks
    5. Reports images processed per second for performance monitoring
    
    The chunking approach allows processing datasets of any size, regardless of
    available RAM, by only keeping a limited number of images in memory at once.
    
    Args:
        image_paths: List of image paths to process
        target_resolution: Tuple of (height, width) for output images
        num_workers: Number of parallel workers (None = auto-detect)
        chunk_size: Size of chunks to process at once (memory usage control)
        use_cache: Whether to use disk caching (True = save preprocessed images)
        
    Returns:
        Number of successfully processed images
        
    Performance considerations:
        - Larger chunk_size = faster processing but more memory usage
        - Higher num_workers = faster processing but more CPU/memory usage
        - Processing time scales nearly linearly with number of images
        - Memory usage is primarily determined by chunk_size * num_workers
    """
    total_images = len(image_paths)
    processed_count = 0
    
    # Process in chunks to avoid memory issues
    for i in range(0, total_images, chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, total_images)
        chunk = image_paths[chunk_start:chunk_end]
        
        chunk_size = len(chunk)
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_images-1)//chunk_size + 1} "
                   f"({chunk_size} images, {chunk_start}-{chunk_end-1}/{total_images-1})")
        
        start_time = time.time()
        
        # Process chunk
        results = preprocess_image_batch(
            chunk, 
            target_resolution=target_resolution,
            num_workers=num_workers, 
            use_cache=use_cache
        )
        
        # Count processed images
        processed_count += len(results)
        
        # Log progress
        elapsed = time.time() - start_time
        images_per_sec = chunk_size / elapsed if elapsed > 0 else 0
        logger.info(f"Processed {len(results)}/{chunk_size} images in {elapsed:.2f}s "
                   f"({images_per_sec:.2f} img/s)")
        
        # Force garbage collection to free memory
        results = None
        gc.collect()
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache all dataset images")
    parser.add_argument("--crohme-dir", type=str, default=CROHME_DATA_PATH,
                       help="Path to CROHME dataset directory")
    parser.add_argument("--mathwriting-dir", type=str, default=MATHWRITING_DATA_PATH,
                       help="Path to MathWriting dataset directory")
    parser.add_argument("--height", type=int, default=512, 
                       help="Target height for preprocessing")
    parser.add_argument("--width", type=int, default=1024, 
                       help="Target width for preprocessing")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers for parallel processing (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Size of chunks to process at once")
    parser.add_argument("--splits", type=str, default="train,val,test",
                       help="Dataset splits to process (comma-separated)")
    parser.add_argument("--disable-cache", action="store_true",
                       help="Disable disk caching (will reprocess all images)")
    parser.add_argument("--dataset", type=str, default="both",
                       choices=["both", "crohme", "mathwriting"],
                       help="Which dataset to process")
    
    args = parser.parse_args()
    
    # Set up parameters
    target_resolution = (args.height, args.width)
    num_workers = args.num_workers
    chunk_size = args.chunk_size
    use_cache = not args.disable_cache
    splits = args.splits.split(',')
    
    # Process specified datasets
    start_time = time.time()
    total_processed = 0
    
    logger.info(f"Starting dataset processing with parameters:")
    logger.info(f"  - Resolution: {args.height}×{args.width}")
    logger.info(f"  - Workers: {num_workers if num_workers else 'auto'}")
    logger.info(f"  - Chunk size: {chunk_size}")
    logger.info(f"  - Cache: {'Enabled' if use_cache else 'Disabled'}")
    logger.info(f"  - Splits: {splits}")
    logger.info(f"  - Datasets: {args.dataset}")
    
    # Process MathWriting dataset
    if args.dataset in ["both", "mathwriting"]:
        logger.info("=== Processing MathWriting dataset ===")
        
        for split in splits:
            logger.info(f"Processing MathWriting {split} split...")
            image_paths = load_dataset_paths(args.mathwriting_dir, "mathwriting", split)
            
            if image_paths:
                processed = process_dataset(
                    image_paths, 
                    target_resolution, 
                    num_workers, 
                    chunk_size, 
                    use_cache
                )
                total_processed += processed
                logger.info(f"Finished processing MathWriting {split}: {processed} images")
            else:
                logger.warning(f"No images found for MathWriting {split}")
    
    # Process CROHME dataset
    if args.dataset in ["both", "crohme"]:
        logger.info("=== Processing CROHME dataset ===")
        
        for split in splits:
            logger.info(f"Processing CROHME {split} split...")
            image_paths = load_dataset_paths(args.crohme_dir, "crohme", split)
            
            if image_paths:
                processed = process_dataset(
                    image_paths, 
                    target_resolution, 
                    num_workers, 
                    chunk_size, 
                    use_cache
                )
                total_processed += processed
                logger.info(f"Finished processing CROHME {split}: {processed} images")
            else:
                logger.warning(f"No images found for CROHME {split}")
    
    # Report overall statistics
    total_time = time.time() - start_time
    images_per_sec = total_processed / total_time if total_time > 0 else 0
    
    logger.info("=== Processing Complete ===")
    logger.info(f"Total processed: {total_processed} images")
    logger.info(f"Total time: {total_time:.2f}s ({images_per_sec:.2f} img/s)")
    logger.info(f"All preprocessed images cached for fast loading")

if __name__ == "__main__":
    main()