#!/usr/bin/env python3
"""
Script to explore a small portion of the datasets using streaming mode.
This avoids downloading the entire dataset at once.
"""

import os
import argparse
import logging
from datasets import load_dataset
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

def explore_dataset(dataset_name, num_examples=5, output_dir=None):
    """
    Explore a small part of a dataset using streaming mode.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        num_examples: Number of examples to look at
        output_dir: Optional directory to save metadata
    """
    logger.info(f"Exploring dataset: {dataset_name}")
    
    try:
        # Load dataset in streaming mode to avoid downloading everything
        dataset = load_dataset(dataset_name, streaming=True)
        
        # Get available splits
        splits = list(dataset.keys())
        logger.info(f"Available splits: {splits}")
        
        # Choose the first split (usually 'train')
        split_name = splits[0]
        split = dataset[split_name]
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving metadata to: {output_path}")
        
        # Look at the first few examples
        logger.info(f"Examining {num_examples} examples from '{split_name}' split:")
        
        examples_metadata = []
        for i, example in enumerate(dataset[split_name].take(num_examples)):
            logger.info(f"\nExample {i+1}:")
            
            # Extract and display example fields
            example_dict = {}
            for key, value in example.items():
                if key == 'image':
                    logger.info(f"  {key}: <Image object>")
                    example_dict[key] = "<Image object>"
                else:
                    example_info = str(value)
                    # Truncate long values
                    if len(example_info) > 100:
                        example_info = example_info[:100] + "..."
                    logger.info(f"  {key}: {example_info}")
                    example_dict[key] = str(value)
            
            examples_metadata.append(example_dict)
        
        # Save metadata if output directory is specified
        if output_dir:
            metadata_file = output_path / f"{dataset_name.replace('/', '_')}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(examples_metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_file}")
        
        logger.info(f"\nDataset exploration complete for {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error exploring dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description="Explore a small portion of a dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to explore (e.g., 'andito/mathwriting-google')")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples to examine")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save metadata")
    
    args = parser.parse_args()
    
    explore_dataset(args.dataset, args.num_examples, args.output_dir)

if __name__ == "__main__":
    main()