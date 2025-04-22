#!/bin/bash
# Script to process CROHME test data with enhanced preprocessing

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Run test with enhanced preprocessing on CROHME test data
echo "Processing CROHME test data with enhanced preprocessing..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  bash -c "
  # Install required dependencies
  pip install pandas tqdm matplotlib datasets --quiet
  
  # Run the processing script
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
import random
import datasets

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

# Try to load CROHME test data from parquet files
print('Loading CROHME test data...')
test_ds = None
test_year = None

try:
    for year in ['2019', '2016', '2014']:
        parquet_path = f'/app/CROHME-full/data/{year}-00000-of-00001.parquet'
        if os.path.exists(parquet_path):
            test_ds = pd.read_parquet(parquet_path)
            test_year = year
            print(f'Loaded CROHME {year} test dataset from parquet file')
            break
except Exception as e:
    print(f'Error loading parquet files: {e}')

# If we couldn't load from parquet, try using the datasets library
if test_ds is None:
    try:
        for year in ['2019', '2016', '2014']:
            try:
                test_ds = datasets.load_dataset('Neeze/CROHME-full', split=f'test_{year}')
                test_year = year
                print(f'Loaded CROHME {year} test dataset from HuggingFace')
                break
            except:
                continue
    except Exception as e:
        print(f'Error loading from HuggingFace: {e}')

if test_ds is None:
    print('Failed to load any CROHME test dataset. Exiting.')
    sys.exit(1)

print(f'Successfully loaded CROHME {test_year} test dataset with {len(test_ds)} samples')

# Sample size (limit for testing)
MAX_SAMPLES = 100
if len(test_ds) > MAX_SAMPLES:
    print(f'Limiting to {MAX_SAMPLES} random samples for testing')
    if isinstance(test_ds, pd.DataFrame):
        test_ds = test_ds.sample(MAX_SAMPLES, random_state=42)
    else:
        indices = random.sample(range(len(test_ds)), MAX_SAMPLES)
        test_ds = test_ds.select(indices)

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

# Results collection for analysis
all_results = []

# Process each image
print(f'\\nProcessing {len(test_ds)} CROHME test images...')
print('=' * 100)
print(f'| {\\\"Image ID\\\":<15} | {\\\"GT LaTeX\\\":<30} | {\\\"Prediction\\\":<30} | {\\\"Score\\\":<7} | {\\\"Match\\\":<5} |')
print('=' * 100)

match_count = 0
for idx, sample in enumerate(tqdm(test_ds)):
    # Extract image and ground truth
    if isinstance(test_ds, pd.DataFrame):
        image = sample['image']
        if isinstance(image, str):
            # Assuming base64 encoding
            import io
            import base64
            image = Image.open(io.BytesIO(base64.b64decode(image)))
        gt_latex = sample['latex']
    else:
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        gt_latex = sample['latex']
    
    # Convert to numpy array if needed
    img_array = np.array(image)
    
    # Apply enhanced preprocessing
    processed = enhanced_preprocessing(img_array)
    
    # Convert to tensor
    tensor = torch.from_numpy(processed).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
    
    # Create mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Save preprocessed image for reference (save just a subset for space)
    if idx < 10:
        output_path = os.path.join(output_dir, f'crohme_{test_year}_{idx}_processed.png')
        cv2.imwrite(output_path, processed.squeeze())
        
        # Save comparison image (original + processed)
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
        plt.savefig(os.path.join(output_dir, f'crohme_{test_year}_{idx}_comparison.png'))
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
        
        # Truncate for display
        gt_display = gt_latex if len(gt_latex) < 27 else gt_latex[:24] + '...'
        latex_display = latex if len(latex) < 27 else latex[:24] + '...'
        
        # Print result (only for first 20 samples to avoid cluttering console)
        if idx < 20:
            print(f'| {idx:<15} | {gt_display:<30} | {latex_display:<30} | {score:<7.4f} | {is_match!s:<5} |')
        
        # Save result
        all_results.append({
            'idx': idx,
            'gt_latex': gt_latex,
            'predicted_latex': latex,
            'score': score,
            'match': is_match
        })
    else:
        if idx < 20:
            print(f'| {idx:<15} | {gt_latex:<30} | No results            | | False |')
        all_results.append({
            'idx': idx,
            'gt_latex': gt_latex,
            'predicted_latex': 'No results',
            'score': 0.0,
            'match': False
        })

# Calculate accuracy
accuracy = match_count / len(test_ds) * 100 if len(test_ds) > 0 else 0
print('=' * 100)
print(f'\\nAccuracy: {accuracy:.2f}% ({match_count}/{len(test_ds)})')

# Save all results to CSV for detailed analysis
import csv
csv_path = os.path.join(output_dir, f'crohme_{test_year}_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['idx', 'gt_latex', 'predicted_latex', 'score', 'match'])
    writer.writeheader()
    writer.writerows(all_results)

print(f'\\nDetailed results saved to {csv_path}')
print('\\nCROHME test processing completed!')
\"
"

echo "Testing completed! Check the 'assets/crohme_results' directory for results."