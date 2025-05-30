#!/bin/bash
# Script to process CROHME dataset from parquet files

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Run test on CROHME
echo "Processing CROHME dataset using parquet files..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  bash -c "
  # Install required dependencies
  pip install pandas pyarrow matplotlib tqdm --quiet

  # Run the script
  python -c \"
import os
import sys
import numpy as np
import torch
from PIL import Image
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import io
import base64
import random

# Add paths
sys.path.append('/app')
sys.path.append('/posformer')

# Import preprocessing function and model components
from Pos_Former.lit_posformer import LitPosFormer
from Pos_Former.datamodule import vocab

# Output directory
output_dir = '/app/assets/crohme_results'
os.makedirs(output_dir, exist_ok=True)

def enhanced_preprocessing(image, target_resolution=(256, 1024)):
    '''Enhanced preprocessing with better stroke visibility and proper inversion'''
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE to improve contrast (helps with faint strokes)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Threshold to get binary image (black strokes on white background)
    _, binary = cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Thicken the strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find content region (using the dilated image)
    coords = cv2.findNonZero(dilated)
    if coords is None or len(coords) == 0:
        # Return properly formatted empty image (black background for CROHME format)
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

# Load PosFormer model
print('Loading PosFormer model...')
checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
lit_model = LitPosFormer.load_from_checkpoint(checkpoint_path, map_location='cpu')
model = lit_model.model
model.eval()
print('Model loaded successfully!')

# Try to load CROHME data from parquet files
print('Loading CROHME data from parquet files...')
test_year = None
test_df = None

for year in ['2019', '2016', '2014']:
    parquet_path = f'/app/CROHME-full/data/{year}-00000-of-00001.parquet'
    if os.path.exists(parquet_path):
        try:
            # Load the parquet file using pandas with pyarrow engine
            test_df = pd.read_parquet(parquet_path, engine='pyarrow')
            test_year = year
            print(f'Successfully loaded CROHME {year} dataset with {len(test_df)} samples')
            break
        except Exception as e:
            print(f'Error loading {year} parquet file: {e}')
            continue

if test_df is None:
    print('Failed to load any CROHME dataset. Exiting.')
    sys.exit(1)

# Limit to a sample size for testing
SAMPLE_SIZE = 10
print(f'Limiting to {SAMPLE_SIZE} random samples for testing')
sample_df = test_df.sample(SAMPLE_SIZE, random_state=42)

# Function to run inference
def run_inference(tensor, mask):
    try:
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
                return [('No results', 0.0)]
    except Exception as e:
        return [(f'Error: {str(e)}', 0.0)]

# Create a summary file
summary_path = os.path.join(output_dir, f'crohme_{test_year}_results.md')
with open(summary_path, 'w') as summary_file:
    summary_file.write(f'# CROHME {test_year} Test Results\\n\\n')
    summary_file.write(f'Processed: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}\\n\\n')
    
    summary_file.write('| # | Ground Truth | Prediction | Score | Match |\\n')
    summary_file.write('|---|-------------|------------|-------|-------|\\n')
    
    # Process each image
    match_count = 0
    processed_count = 0
    
    print(f'\\nProcessing {len(sample_df)} CROHME test images...')
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        try:
            # Extract the image and ground truth
            image_data = row['image']
            gt_latex = row['latex']
            
            # Handle base64 encoded images
            if isinstance(image_data, str):
                try:
                    # Try to decode base64
                    img_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(img_bytes))
                    img_array = np.array(image)
                except:
                    print(f'Error decoding image for sample {idx}')
                    continue
            else:
                # Direct numpy array
                img_array = np.array(image_data)
            
            # Check if image is valid
            if img_array.size == 0:
                print(f'Empty image for sample {idx}')
                continue
                
            # Apply enhanced preprocessing
            processed = enhanced_preprocessing(img_array)
            
            # Convert to tensor
            tensor = torch.from_numpy(processed).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
            
            # Create mask
            mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
            
            # Save processed image
            output_path = os.path.join(output_dir, f'crohme_{test_year}_{processed_count}_processed.png')
            cv2.imwrite(output_path, processed.squeeze())
            
            # Save comparison image
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img_array, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(processed.squeeze(), cmap='gray')
            plt.title('Processed')
            plt.axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, f'crohme_{test_year}_{processed_count}_comparison.png')
            plt.savefig(comparison_path)
            plt.close()
            
            # Run inference
            results = run_inference(tensor, mask)
            
            if results:
                # Get best result
                latex, score = results[0]
                
                # Check for exact match
                is_match = (latex.strip() == gt_latex.strip())
                if is_match:
                    match_count += 1
                
                # Add to summary
                gt_display = gt_latex.replace('|', '\\|') if len(gt_latex) < 30 else gt_latex[:27].replace('|', '\\|') + '...'
                pred_display = latex.replace('|', '\\|') if len(latex) < 30 else latex[:27].replace('|', '\\|') + '...'
                
                summary_file.write(f'| {processed_count} | `{gt_display}` | `{pred_display}` | {score:.4f} | {is_match} |\\n')
                
                # Save detailed results
                with open(os.path.join(output_dir, f'crohme_{test_year}_{processed_count}_results.txt'), 'w') as f:
                    f.write(f'Sample ID: {idx}\\n')
                    f.write(f'GT LaTeX: {gt_latex}\\n\\n')
                    
                    for i, (result_latex, result_score) in enumerate(results):
                        f.write(f'Result {i+1} (score: {result_score:.4f}):\\n')
                        f.write(f'{result_latex}\\n\\n')
            else:
                summary_file.write(f'| {processed_count} | `{gt_latex}` | No results | - | False |\\n')
            
            processed_count += 1
            
        except Exception as e:
            print(f'Error processing sample {idx}: {e}')
    
    # Add summary statistics
    accuracy = match_count / processed_count * 100 if processed_count > 0 else 0
    
    summary_file.write(f'\\n\\n## Summary Statistics\\n\\n')
    summary_file.write(f'- Total samples processed: {processed_count}\\n')
    summary_file.write(f'- Exact matches: {match_count}\\n')
    summary_file.write(f'- Accuracy: {accuracy:.2f}%\\n')

print(f'\\nProcessed {processed_count} samples with {match_count} exact matches')
print(f'Accuracy: {match_count/processed_count*100:.2f}% ({match_count}/{processed_count})')
print(f'\\nResults saved to {summary_path}')
\"
"

echo "Testing completed! Check the 'assets/crohme_results' directory for results."