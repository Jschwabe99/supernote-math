#!/usr/bin/env python3
"""
Explore all splits of a dataset to understand its structure.
This script uses a more targeted approach to examine splits without streaming.
"""

import os
import sys
import argparse
import logging
from datasets import load_dataset_builder
from pathlib import Path
import json

# Set extremely high timeout for dataset operations (effectively no timeout)
os.environ["HF_DATASETS_TIMEOUT"] = str(24 * 60 * 60)  # 24 hours in seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def explore_dataset_info(dataset_name, output_dir=None):
    """
    Explore dataset structure by examining its splits and features.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        output_dir: Optional directory to save metadata
    """
    logger.info(f"Exploring dataset structure: {dataset_name}")
    
    try:
        # Get dataset builder info without downloading the full dataset
        builder = load_dataset_builder(dataset_name)
        
        # Extract info about the dataset
        info = {
            "name": dataset_name,
            "description": builder.info.description[:200] + "..." if len(builder.info.description) > 200 else builder.info.description,
            "splits": {},
            "features": str(builder.info.features),
            "download_size_mb": builder.info.download_size / (1024 * 1024) if builder.info.download_size else None,
            "dataset_size_mb": builder.info.dataset_size / (1024 * 1024) if builder.info.dataset_size else None,
        }
        
        # Get info for each split
        for split_name, split_info in builder.info.splits.items():
            info["splits"][split_name] = {
                "num_examples": split_info.num_examples,
                "size_mb": split_info.num_bytes / (1024 * 1024) if split_info.num_bytes else None,
            }
        
        # Print the dataset information
        logger.info(f"Dataset: {info['name']}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"Download size: {info['download_size_mb']:.2f} MB" if info['download_size_mb'] else "Download size: Unknown")
        logger.info(f"Dataset size: {info['dataset_size_mb']:.2f} MB" if info['dataset_size_mb'] else "Dataset size: Unknown")
        logger.info(f"Features: {info['features']}")
        
        # Print splits information
        logger.info("Splits:")
        for split_name, split_info in info["splits"].items():
            size_str = f"{split_info['size_mb']:.2f} MB" if split_info['size_mb'] else "Unknown"
            logger.info(f"  {split_name}: {split_info['num_examples']} examples, {size_str}")
        
        # Save dataset info if output directory is specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Save dataset info to JSON file
            info_file = output_path / f"{dataset_name.replace('/', '_')}_info.json"
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Saved dataset info to {info_file}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error exploring dataset: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Explore dataset structure")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to explore (e.g., 'andito/mathwriting-google')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save info")
    
    args = parser.parse_args()
    
    explore_dataset_info(args.dataset, args.output_dir)

if __name__ == "__main__":
    main()