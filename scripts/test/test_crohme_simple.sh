#!/bin/bash
# Simple script to test on CROHME samples directly from parquet files

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Run test on CROHME
echo "Testing on CROHME dataset samples..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  bash -c "
  # Install required dependencies
  pip install pandas matplotlib --quiet

  # Run the script
  python -c \"
import os
import sys
import numpy as np
import torch
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

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

# Try to load CROHME data directly from parquet files
print('Checking for CROHME parquet files...')
test_year = None
parquet_path = None

for year in ['2019', '2016', '2014']:
    path = f'/app/CROHME-full/data/{year}-00000-of-00001.parquet'
    if os.path.exists(path):
        parquet_path = path
        test_year = year
        print(f'Found CROHME {year} parquet file')
        break

if parquet_path is None:
    print('No CROHME parquet files found. Exiting.')
    sys.exit(1)

# Load a few samples from the parquet file
print(f'Loading samples from {parquet_path}...')
try:
    # Read just 5 samples for quick testing
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    # Take first 5 rows
    sample_df = df.head(5)
    print(f'Successfully loaded {len(sample_df)} samples')
except Exception as e:
    print(f'Error loading parquet file: {e}')
    sys.exit(1)

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

# Process each image in the sample
print('\\nProcessing CROHME samples...')
print('=' * 80)

match_count = 0
results_summary = []

for idx, row in enumerate(sample_df.itertuples()):
    try:
        # Get the image and ground truth
        img_data = row.image
        gt_latex = row.latex
        
        # Handle base64 encoded images
        if isinstance(img_data, str):
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(image)
        else:
            img_array = np.array(img_data)
        
        print(f'Processing image {idx+1}/{len(sample_df)}...')
        
        # Apply enhanced preprocessing
        processed = enhanced_preprocessing(img_array)
        
        # Save processed image
        output_path = os.path.join(output_dir, f'crohme_{test_year}_sample_{idx}.png')
        cv2.imwrite(output_path, processed.squeeze())
        
        # Convert to tensor for model
        tensor = torch.from_numpy(processed).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
        
        # Create mask
        mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
        
        # Run inference
        results = run_inference(tensor, mask)
        
        if results:
            # Get best result
            latex, score = results[0]
            
            # Check for exact match
            is_match = (latex.strip() == gt_latex.strip())
            if is_match:
                match_count += 1
            
            print(f'  GT LaTeX: {gt_latex}')
            print(f'  Prediction: {latex}')
            print(f'  Score: {score:.4f}')
            print(f'  Match: {is_match}')
            print('  ' + '-' * 60)
            
            # Save result
            results_summary.append({
                'idx': idx,
                'gt_latex': gt_latex,
                'predicted_latex': latex,
                'score': score,
                'match': is_match
            })
        else:
            print(f'  No results for image {idx}')
    except Exception as e:
        print(f'  Error processing image {idx}: {e}')

# Calculate accuracy
accuracy = match_count / len(sample_df) * 100 if len(sample_df) > 0 else 0
print('=' * 80)
print(f'\\nAccuracy: {accuracy:.2f}% ({match_count}/{len(sample_df)})')

# Save summary
summary_path = os.path.join(output_dir, f'crohme_{test_year}_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f'# CROHME {test_year} Test Results\\n')
    f.write(f'Test date: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}\\n\\n')
    f.write(f'Sample size: {len(sample_df)} images\\n')
    f.write(f'Exact matches: {match_count}\\n')
    f.write(f'Accuracy: {accuracy:.2f}%\\n\\n')
    
    f.write('Summary of results:\\n\\n')
    
    for result in results_summary:
        f.write(f\"Image {result['idx']}:\\n\")
        f.write(f\"  GT: {result['gt_latex']}\\n\")
        f.write(f\"  Pred: {result['predicted_latex']}\\n\")
        f.write(f\"  Score: {result['score']:.4f}\\n\")
        f.write(f\"  Match: {result['match']}\\n\\n\")

print(f'\\nSummary saved to {summary_path}')
print('CROHME test completed!')
\"
"

echo "Testing completed! Check the 'assets/crohme_results' directory for results."