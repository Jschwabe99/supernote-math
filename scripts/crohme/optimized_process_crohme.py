#!/usr/bin/env python3
"""
Script to process CROHME dataset and evaluate the PosFormer model - Optimized version
"""
import os
import sys
import numpy as np
import torch
from PIL import Image
import pandas as pd
import time
import cv2
from tqdm import tqdm
import io
import base64
import random
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Parse arguments
parser = argparse.ArgumentParser(description='Process CROHME dataset')
parser.add_argument('--posformer_dir', type=str, default='../PosFormer-main',
                    help='Path to PosFormer directory')
parser.add_argument('--crohme_dir', type=str, default='CROHME-full',
                    help='Path to CROHME-full directory')
parser.add_argument('--output_dir', type=str, default='assets/crohme_results',
                    help='Output directory for results')
parser.add_argument('--sample_size', type=int, default=100,
                    help='Number of samples to process')
parser.add_argument('--year', type=str, default=None,
                    help='Specific year to test (2014, 2016, 2019)')
parser.add_argument('--no_invert', action='store_true',
                    help='Do not invert images (use for CROHME dataset)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker threads (0 for auto)')
args = parser.parse_args()

# Convert relative paths to absolute paths
if not os.path.isabs(args.posformer_dir):
    args.posformer_dir = os.path.abspath(os.path.join(parent_dir, args.posformer_dir))
if not os.path.isabs(args.crohme_dir):
    args.crohme_dir = os.path.abspath(os.path.join(parent_dir, args.crohme_dir))
if not os.path.isabs(args.output_dir):
    args.output_dir = os.path.abspath(os.path.join(parent_dir, args.output_dir))

# Add PosFormer path
sys.path.append(args.posformer_dir)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Now import PosFormer components
try:
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.datamodule import vocab
except ImportError as e:
    print(f"Error importing PosFormer components: {e}")
    print(f"Make sure the PosFormer directory is correct: {args.posformer_dir}")
    sys.exit(1)

def enhanced_preprocessing(image, target_resolution=(256, 1024), is_crohme=False):
    """
    Enhanced preprocessing with better stroke visibility and proper inversion if needed
    
    Args:
        image: Input image as numpy array
        target_resolution: Target output resolution (height, width)
        is_crohme: Whether the image is from CROHME dataset (already white-on-black)
    
    Returns:
        Processed image as numpy array with shape (height, width, 1)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE to improve contrast (helps with faint strokes)
    # Use LRU cache for CLAHE object to avoid recreating it
    @lru_cache(maxsize=1)
    def get_clahe():
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    clahe = get_clahe()
    enhanced = clahe.apply(gray)
    
    # Handle different dataset formats
    if is_crohme:
        # CROHME is already white on black, just threshold to make it binary
        _, binary = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)
    else:
        # Supernote handwriting is black on white, invert to get white on black
        _, binary = cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Thicken the strokes - cache the kernel for performance
    @lru_cache(maxsize=1)
    def get_kernel():
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    kernel = get_kernel()
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find content region (using the dilated image)
    coords = cv2.findNonZero(dilated)
    if coords is None or len(coords) == 0:
        # Return properly formatted empty image (black background)
        return np.zeros((*target_resolution, 1), dtype=np.uint8)
    
    # Add margin and crop
    x, y, w, h = cv2.boundingRect(coords)
    margin = max(20, int(max(w, h) * 0.05))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(dilated.shape[1], x + w + margin)
    y2 = min(dilated.shape[0], y + h + margin)
    cropped = dilated[y1:y2, x1:x2]
    
    # Scale to target size
    target_h, target_w = target_resolution
    h, w = cropped.shape
    
    # Preserve aspect ratio
    aspect = w / h if h > 0 else 1.0
    target_aspect = target_w / target_h
    
    if aspect > target_aspect:
        new_w = target_w
        new_h = min(target_h, int(target_w / aspect))
    else:
        new_h = target_h
        new_w = min(target_w, int(target_h * aspect))
    
    # Resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas (black background) and center content (white strokes)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Add channel dimension and return
    # The format is now: white strokes (255) on black background (0)
    # This matches the CROHME dataset format expected by the model
    return canvas.reshape(target_h, target_w, 1)

def run_inference(model, tensor, mask):
    """Run inference with the model using GPU if available"""
    try:
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            tensor = tensor.to(device)
            mask = mask.to(device)
            model = model.to(device)
        
        with torch.no_grad():
            hypotheses = model.beam_search(
                tensor,
                mask,
                beam_size=8,          # Larger beam size
                max_len=50,           # Longer sequences
                alpha=1.0,
                early_stopping=True,
                temperature=1.0
            )
            
            if hypotheses:
                results = []
                for i, hyp in enumerate(hypotheses[:3]):  # Get top 3 results
                    latex = vocab.indices2label(hyp.seq)
                    score = float(hyp.score)
                    results.append((latex, score))
                return results
            else:
                return [("No results", 0.0)]
    except Exception as e:
        return [(f"Error: {str(e)}", 0.0)]

# Use a cache system for image extraction to avoid processing the same image multiple times
@lru_cache(maxsize=32)
def extract_image_from_dict(image_dict_id):
    """
    Extract image from dictionary format used in CROHME parquet files
    The dict typically has format like:
    {
        'bytes': b'...',  # The actual image data
        'path': '...',    # Optional path
        'height': 123,    # Optional height
        'width': 456      # Optional width
    }
    
    Note: We use an ID for caching since the dict itself isn't hashable
    """
    # Get the image dict from the global cache
    global image_dict_cache
    if image_dict_id not in image_dict_cache:
        return None
        
    image_dict = image_dict_cache[image_dict_id]
    
    if not isinstance(image_dict, dict):
        return None
        
    # Try the 'bytes' key first (most common)
    if 'bytes' in image_dict:
        img_bytes = image_dict['bytes']
        if isinstance(img_bytes, bytes):
            try:
                img = Image.open(io.BytesIO(img_bytes))
                return np.array(img)
            except:
                pass
                
    # Try base64 encoded bytes
    if 'bytes' in image_dict and isinstance(image_dict['bytes'], str):
        try:
            img_bytes = base64.b64decode(image_dict['bytes'])
            img = Image.open(io.BytesIO(img_bytes))
            return np.array(img)
        except:
            pass
            
    # If there's a 'path' key and it exists, try loading from that
    if 'path' in image_dict and os.path.exists(image_dict['path']):
        try:
            img = Image.open(image_dict['path'])
            return np.array(img)
        except:
            pass
    
    # If there are 'array' or 'data' keys, try those
    for key in ['array', 'data']:
        if key in image_dict:
            try:
                return np.array(image_dict[key])
            except:
                pass
                
    # Last resort - try to interpret the entire dict as a numpy array
    try:
        # This works if the dict is actually a serialized numpy array
        return np.array(list(image_dict.values()))
    except:
        pass
        
    # If we got here, we couldn't extract the image
    return None

def main():
    # Global cache for image dicts since they're not hashable
    global image_dict_cache
    image_dict_cache = {}
    
    # Load PosFormer model
    print("Loading PosFormer model...")
    checkpoint_path = os.path.join(args.posformer_dir, "lightning_logs/version_0/checkpoints/best.ckpt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Detect if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
        
    try:
        lit_model = LitPosFormer.load_from_checkpoint(checkpoint_path, map_location=device)
        model = lit_model.model
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Try to load CROHME data from parquet files
    print("Loading CROHME data from parquet files...")
    test_year = args.year
    test_df = None

    # If specific year is provided, try that first
    if test_year:
        years_to_try = [test_year]
    else:
        years_to_try = ["2019", "2016", "2014"]
    
    # Set is_crohme flag based on command line or source
    is_crohme = args.no_invert or "CROHME" in args.crohme_dir
    if is_crohme:
        print("Processing as CROHME dataset (no inversion needed)")
        
    # Based on README.md, columns are "image" and "label"
    for year in years_to_try:
        parquet_path = os.path.join(args.crohme_dir, f"data/{year}-00000-of-00001.parquet")
        if os.path.exists(parquet_path):
            try:
                # Load the parquet file using pandas with pyarrow engine
                test_df = pd.read_parquet(parquet_path, engine="pyarrow")
                test_year = year
                print(f"Successfully loaded CROHME {year} dataset with {len(test_df)} samples")
                
                # Display dataframe column information
                print("Dataframe columns:", test_df.columns.tolist())
                print("Column types:")
                for col in test_df.columns:
                    print(f"  {col}: {test_df[col].dtype}")
                    
                break
            except Exception as e:
                print(f"Error loading {year} parquet file: {e}")
                continue

    if test_df is None:
        print("Failed to load any CROHME dataset. Exiting.")
        sys.exit(1)

    # Ensure the expected columns are present
    if "image" not in test_df.columns:
        print("Error: 'image' column not found in the dataframe")
        sys.exit(1)
        
    # Check if the label column is "label" or "latex"
    label_column = "label" if "label" in test_df.columns else "latex"
    if label_column not in test_df.columns:
        print(f"Error: Neither 'label' nor 'latex' column found in the dataframe")
        sys.exit(1)

    print(f"Using '{label_column}' column for ground truth labels")

    # Limit to a sample size for testing
    print(f"Limiting to {args.sample_size} random samples for testing")
    sample_df = test_df.sample(args.sample_size, random_state=42)
    
    # Put image dicts into the global cache
    print("Preparing image cache...")
    for idx, row in sample_df.iterrows():
        if isinstance(row["image"], dict):
            # Use the index as a key
            image_dict_cache[idx] = row["image"]

    # Create a summary file
    summary_path = os.path.join(args.output_dir, f"crohme_{test_year}_summary.md")
    with open(summary_path, "w") as summary_file:
        # Write header
        summary_file.write(f"# CROHME {test_year} Test Results\n\n")
        summary_file.write(f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary_file.write("| # | Ground Truth | Prediction | Score | Match |\n")
        summary_file.write("|---|-------------|------------|-------|-------|\n")
        
        # Process each image using parallel processing
        results_list = []
        
        # Function to process a single sample
        def process_sample(idx, row):
            try:
                # Extract the image and ground truth
                image_data = row["image"]
                gt_latex = row[label_column]
                
                # Handle different image formats
                img_array = None
                
                # Try different approaches to handle the image data
                if isinstance(image_data, dict):
                    # Use dict from the cache
                    img_array = extract_image_from_dict(idx)
                elif isinstance(image_data, bytes):
                    # Direct bytes - try to load as PIL Image
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        img_array = np.array(img)
                    except:
                        return None
                elif isinstance(image_data, str):
                    # Could be base64 encoded
                    try:
                        img_bytes = base64.b64decode(image_data)
                        img = Image.open(io.BytesIO(img_bytes))
                        img_array = np.array(img)
                    except:
                        return None
                elif isinstance(image_data, (np.ndarray, list)):
                    # Direct array
                    img_array = np.array(image_data)
                else:
                    return None
                
                # Check if image is valid
                if img_array is None or img_array.size == 0:
                    return None
                    
                # Apply enhanced preprocessing (without inversion for CROHME)
                processed = enhanced_preprocessing(img_array, is_crohme=is_crohme)
                
                # Convert to tensor
                tensor = torch.from_numpy(processed).float() / 255.0
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
                
                # Create mask
                mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device="cpu")
                
                # Run inference
                results = run_inference(model, tensor, mask)
                
                if results:
                    # Get best result
                    latex, score = results[0]
                    
                    # Check for exact match
                    is_match = (latex.strip() == gt_latex.strip())
                    
                    return {
                        'idx': idx,
                        'gt_latex': gt_latex,
                        'pred_latex': latex,
                        'score': score,
                        'is_match': is_match
                    }
                else:
                    return {
                        'idx': idx, 
                        'gt_latex': gt_latex,
                        'pred_latex': "No results",
                        'score': 0.0,
                        'is_match': False
                    }
                    
            except Exception as e:
                return {
                    'idx': idx,
                    'error': str(e)
                }
        
        print(f"\nProcessing {len(sample_df)} CROHME test images in parallel...")
        
        # Use ThreadPoolExecutor for parallel processing
        # Determine number of workers based on CPU cores
        if args.num_workers <= 0:
            num_workers = min(os.cpu_count() or 4, 8)  # Use at most 8 workers by default
        else:
            num_workers = args.num_workers
            
        print(f"Using {num_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create tasks for each sample
            future_to_idx = {
                executor.submit(process_sample, idx, row): idx 
                for idx, row in sample_df.iterrows()
            }
            
            # Process results as they complete
            for future in tqdm(future_to_idx, total=len(sample_df)):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_list.append(result)
                except Exception as e:
                    print(f"Sample {idx} processing failed: {e}")
        
        # Sort results by index
        results_list.sort(key=lambda x: x.get('idx', 0))
        
        # Count matches
        match_count = sum(1 for r in results_list if r.get('is_match', False))
        processed_count = len(results_list)
        
        # Write results to summary file
        for i, result in enumerate(results_list):
            if 'error' in result:
                summary_file.write(f"| {i} | Error | {result['error']} | - | False |\n")
                continue
                
            gt_display = result['gt_latex']
            if len(gt_display) > 30:
                gt_display = gt_display[:27] + "..."
            gt_display = gt_display.replace("|", "\\|")
            
            pred_display = result['pred_latex']
            if len(pred_display) > 30:
                pred_display = pred_display[:27] + "..."
            pred_display = pred_display.replace("|", "\\|")
            
            summary_file.write(f"| {i} | `{gt_display}` | `{pred_display}` | {result.get('score', 0):.4f} | {result.get('is_match', False)} |\n")
        
        # Add summary statistics
        accuracy = match_count / processed_count * 100 if processed_count > 0 else 0
        
        summary_file.write(f"\n\n## Summary Statistics\n\n")
        summary_file.write(f"- Total samples processed: {processed_count}\n")
        summary_file.write(f"- Exact matches: {match_count}\n")
        summary_file.write(f"- Accuracy: {accuracy:.2f}%\n")
        summary_file.write(f"- Device used: {device}\n")
        summary_file.write(f"- Workers: {num_workers}\n")

    print(f"\nProcessed {processed_count} samples with {match_count} exact matches")
    if processed_count > 0:
        print(f"Accuracy: {match_count/processed_count*100:.2f}% ({match_count}/{processed_count})")
    else:
        print("No samples processed")
    print(f"\nResults saved to {summary_path}")

if __name__ == "__main__":
    main()