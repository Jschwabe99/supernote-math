#!/bin/bash
# Script to test on a sample of CROHME directly using HuggingFace datasets

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Run test on CROHME
echo "Testing on CROHME dataset..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  bash -c "
  # Install required dependencies
  pip install datasets tqdm matplotlib --quiet

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
from tqdm import tqdm
import random
from datasets import load_dataset

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

# Try to load CROHME data directly
print('Loading CROHME dataset...')
try:
    # Try loading a sample of test data from HuggingFace
    print('Trying to load CROHME from HuggingFace...')
    crohme_ds = load_dataset('Neeze/CROHME-full', split='test_2019')
    test_year = '2019'
    print(f'Successfully loaded CROHME {test_year} test dataset with {len(crohme_ds)} samples')
except Exception as e:
    print(f'Failed to load from HuggingFace: {e}')
    try:
        # Try loading a reduced subset instead as a fallback
        crohme_ds = load_dataset('Neeze/CROHME', split='test')
        test_year = 'subset'
        print(f'Successfully loaded CROHME subset with {len(crohme_ds)} samples')
    except Exception as e:
        print(f'Failed to load reduced subset: {e}')
        print('Cannot load CROHME dataset. Exiting.')
        sys.exit(1)

# Limit to a sample size for testing
sample_size = 10  # Small sample for quick testing
print(f'Limiting to a sample of {sample_size} images')
indices = random.sample(range(len(crohme_ds)), sample_size)
sample_ds = crohme_ds.select(indices)

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
print('\\nProcessing CROHME test images...')
print('=' * 100)
print(f'| {\"Image ID\":<15} | {\"GT LaTeX\":<30} | {\"Prediction\":<30} | {\"Score\":<7} | {\"Match\":<5} |')
print('=' * 100)

match_count = 0
all_results = []

for idx, sample in enumerate(sample_ds):
    # Extract image and ground truth
    image = sample['image']
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    gt_latex = sample['latex']
    
    # Apply enhanced preprocessing
    processed = enhanced_preprocessing(image)
    
    # Convert to tensor
    tensor = torch.from_numpy(processed).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
    
    # Create mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Save preprocessed image
    output_path = os.path.join(output_dir, f'crohme_{test_year}_{idx}_processed.png')
    cv2.imwrite(output_path, processed.squeeze())
    
    # Save comparison image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
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
        
        # Print result
        print(f'| {idx:<15} | {gt_display:<30} | {latex_display:<30} | {score:<7.4f} | {is_match!s:<5} |')
        
        # Save result
        all_results.append({
            'idx': idx,
            'gt_latex': gt_latex,
            'predicted_latex': latex,
            'score': score,
            'match': is_match
        })
        
        # Save detailed results
        with open(os.path.join(output_dir, f'crohme_{test_year}_{idx}_results.txt'), 'w') as f:
            f.write(f'GT LaTeX: {gt_latex}\\n\\n')
            for i, (result_latex, result_score) in enumerate(results):
                f.write(f'Result {i+1} (score: {result_score:.4f}):\\n')
                f.write(f'{result_latex}\\n\\n')
    else:
        print(f'| {idx:<15} | {gt_display:<30} | No results | | False |')

# Calculate accuracy
accuracy = match_count / len(sample_ds) * 100
print('=' * 100)
print(f'\\nAccuracy: {accuracy:.2f}% ({match_count}/{len(sample_ds)})')

# Save summary
with open(os.path.join(output_dir, f'crohme_{test_year}_summary.txt'), 'w') as f:
    f.write(f'# CROHME {test_year} Test Results\\n')
    f.write(f'Test date: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}\\n\\n')
    f.write(f'Sample size: {len(sample_ds)} images\\n')
    f.write(f'Exact matches: {match_count}\\n')
    f.write(f'Accuracy: {accuracy:.2f}%\\n\\n')
    
    f.write('| Image ID | GT LaTeX | Prediction | Match |\\n')
    f.write('|----------|----------|------------|-------|\\n')
    
    for result in all_results:
        f.write(f\"| {result['idx']} | {result['gt_latex']} | {result['predicted_latex']} | {result['match']} |\\n\")

print('\\nCROHME test completed! Results saved to the output directory.')
\"
"

echo "Testing completed! Check the 'assets/crohme_results' directory for results."