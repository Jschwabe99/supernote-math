#!/bin/bash
# Script to test the enhanced preprocessing method

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/enhanced_results"

# Run test with enhanced preprocessing
echo "Testing enhanced preprocessing method..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  python -c "
import os
import sys
import numpy as np
import torch
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
import time

# Add paths
sys.path.append('/app')
sys.path.append('/posformer')

# Import preprocessing function and model components
from Pos_Former.lit_posformer import LitPosFormer
from Pos_Former.datamodule import vocab

# Output directory
output_dir = '/app/assets/enhanced_results'
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

# Find all test page PNGs
page_images = glob.glob('/app/assets/20250422_073317_Page_*.png')
page_images.sort()

# Add Test_Data image if it exists
test_data_path = '/app/assets/Test_Data_Page_1.png'
if os.path.exists(test_data_path):
    page_images.append(test_data_path)

print(f'Found {len(page_images)} test page images')

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

# Process each image with enhanced preprocessing
print('\\n' + '=' * 100)
print(f'| {\"Image\":<25} | {\"LaTeX (Enhanced Preprocessing)\":<65} | {\"Score\":<7} |')
print('=' * 100)

for image_path in page_images:
    basename = os.path.basename(image_path)
    print(f'Processing: {basename}')
    
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Apply enhanced preprocessing
    processed = enhanced_preprocessing(img_array)
    
    # Convert to tensor
    tensor = torch.from_numpy(processed).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
    
    # Create mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Save preprocessed image for visual reference
    output_path = os.path.join(output_dir, f'{os.path.splitext(basename)[0]}_enhanced.png')
    cv2.imwrite(output_path, processed.squeeze())
    
    # Run inference
    start_time = time.time()
    results = run_inference(tensor, mask)
    process_time = time.time() - start_time
    
    if results:
        # Get best result
        latex, score = results[0]
        
        # Truncate for display
        latex_display = latex if len(latex) < 62 else latex[:59] + '...'
        
        # Print result
        print(f'| {basename:<25} | {latex_display:<65} | {score:<7.4f} |')
        
        # Save all results to text file
        with open(os.path.join(output_dir, f'{os.path.splitext(basename)[0]}_results.txt'), 'w') as f:
            f.write(f'Image: {basename}\\n')
            f.write(f'Processing time: {process_time:.2f}s\\n\\n')
            
            for i, (result_latex, result_score) in enumerate(results):
                f.write(f'Result {i+1} (score: {result_score:.4f}):\\n')
                f.write(f'{result_latex}\\n\\n')
    else:
        print(f'| {basename:<25} | No results | |')
    
    print('-' * 100)

print('\\nAll images processed with enhanced preprocessing!')
"

echo "Testing completed! Check the 'assets/enhanced_results' directory for results."