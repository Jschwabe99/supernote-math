#!/bin/bash
# Script to test enhanced preprocessing for better recognition

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create visualization output directory
mkdir -p "$DIR/assets/enhanced_preprocessing"

# Run enhanced preprocessing script
echo "Testing enhanced preprocessing approaches..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer \
  python -c "
import os
import sys
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2

# Add paths
sys.path.append('/app')
sys.path.append('/posformer')

# Import preprocessing function
from core.data.loaders import preprocess_image
from Pos_Former.lit_posformer import LitPosFormer
from Pos_Former.datamodule import vocab

# Output directory
output_dir = '/app/assets/enhanced_preprocessing'
os.makedirs(output_dir, exist_ok=True)

def enhanced_preprocessing(image, target_resolution=(256, 1024)):
    '''Enhanced preprocessing with better stroke visibility'''
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Thicken the strokes
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Invert to get white on black (CROHME format)
    inverted = 255 - dilated
    
    # Find content region
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) == 0:
        return np.zeros((*target_resolution, 1), dtype=np.uint8)
    
    # Add margin and crop
    x, y, w, h = cv2.boundingRect(coords)
    margin = max(20, int(max(w, h) * 0.05))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(inverted.shape[1], x + w + margin)
    y2 = min(inverted.shape[0], y + h + margin)
    cropped = inverted[y1:y2, x1:x2]
    
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
    
    # Create canvas and center content
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Add channel dimension
    return canvas.reshape(target_h, target_w, 1)

# Load PosFormer model
print('Loading PosFormer model...')
checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
lit_model = LitPosFormer.load_from_checkpoint(checkpoint_path, map_location='cpu')
model = lit_model.model
model.eval()
print('Model loaded successfully!')

# Find test pages
page_images = glob.glob('/app/assets/20250422_073317_Page_*.png')
page_images.sort()

# Process each image with various preprocessing methods
for idx, image_path in enumerate(page_images[:3]):  # Process first 3 images
    basename = os.path.basename(image_path)
    print(f'Processing {basename}...')
    
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Apply different preprocessing methods
    standard = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=False,
        normalize=False
    ).squeeze()
    
    inverted = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=True,
        normalize=False
    ).squeeze()
    
    enhanced = enhanced_preprocessing(img_array).squeeze()
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original
    axes[0, 0].imshow(img_array, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Standard
    axes[0, 1].imshow(standard, cmap='gray')
    axes[0, 1].set_title('Standard (invert=False)')
    axes[0, 1].axis('off')
    
    # Inverted
    axes[1, 0].imshow(inverted, cmap='gray')
    axes[1, 0].set_title('Inverted (invert=True)')
    axes[1, 0].axis('off')
    
    # Enhanced
    axes[1, 1].imshow(enhanced, cmap='gray')
    axes[1, 1].set_title('Enhanced Method')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Preprocessing Comparison: {basename}')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{os.path.splitext(basename)[0]}_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f'  Saved comparison to {output_path}')
    
    # Run inference with each preprocessing method
    print('  Running inference with different preprocessing methods:')
    
    # Function to run inference
    def run_inference(preprocessed_img, method_name):
        # Convert to tensor
        tensor = torch.from_numpy(preprocessed_img).float() / 255.0
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWCB to BCHW
            
        # Create mask
        mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
        
        # Run beam search
        try:
            hypotheses = model.beam_search(
                tensor,
                mask,
                beam_size=5,
                max_len=30,
                alpha=1.0,
                early_stopping=True,
                temperature=1.0
            )
            
            if hypotheses:
                return vocab.indices2label(hypotheses[0].seq)
            else:
                return 'No results'
        except Exception as e:
            return f'Error: {str(e)}'
    
    # Run with different methods
    standard_result = run_inference(standard.reshape(256, 1024, 1), 'Standard')
    inverted_result = run_inference(inverted.reshape(256, 1024, 1), 'Inverted')
    enhanced_result = run_inference(enhanced.reshape(256, 1024, 1), 'Enhanced')
    
    # Print results
    print(f'    - Standard: {standard_result[:60]}...' if len(standard_result) > 60 else f'    - Standard: {standard_result}')
    print(f'    - Inverted: {inverted_result[:60]}...' if len(inverted_result) > 60 else f'    - Inverted: {inverted_result}')
    print(f'    - Enhanced: {enhanced_result[:60]}...' if len(enhanced_result) > 60 else f'    - Enhanced: {enhanced_result}')
    
    # Save results to text file
    with open(os.path.join(output_dir, f'{os.path.splitext(basename)[0]}_results.txt'), 'w') as f:
        f.write(f'Standard: {standard_result}\\n\\n')
        f.write(f'Inverted: {inverted_result}\\n\\n')
        f.write(f'Enhanced: {enhanced_result}\\n\\n')

print('\\nEnhanced preprocessing testing completed!')
"

echo "Testing completed! Check the 'assets/enhanced_preprocessing' directory for results."