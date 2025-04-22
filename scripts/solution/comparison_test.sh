#!/bin/bash
# Test script to compare preprocessing for different image types

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run comparison test
echo "Running preprocessing comparison test..."
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
import torch
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import io
import base64

# Add paths
sys.path.append('/posformer')
sys.path.append('/app')

# Import components
try:
    from core.data.loaders import preprocess_image
    
    # Create output directory for visualizations
    output_dir = '/app/assets/preprocessing_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    def preprocess_and_save(image_path, output_filename, invert=False):
        # Load image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Preprocess image
        processed = preprocess_image(
            image=img_array,
            target_resolution=(256, 1024),
            invert=invert,
            normalize=False
        )
        
        # Squeeze to 2D
        processed_2d = processed.squeeze()
        
        # Create figure with original and processed images
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original image
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(processed_2d, cmap='gray')
        axes[1].set_title(f'Processed (invert={invert})')
        axes[1].axis('off')
        
        # Add image info
        plt.suptitle(f'Image: {os.path.basename(image_path)}')
        plt.figtext(0.02, 0.02, f'Shape: {img_array.shape} → {processed_2d.shape}, Min/Max: {processed_2d.min()}/{processed_2d.max()}')
        
        # Save figure
        output_path = os.path.join(output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    # Find a CROHME image
    crohme_images = glob.glob('/app/assets/test_output_*/CROHME_hf_*_comparison.png')
    if crohme_images:
        crohme_img = crohme_images[0]
        print(f'Processing CROHME image: {os.path.basename(crohme_img)}')
        
        # Process with and without inversion
        crohme_standard = preprocess_and_save(crohme_img, 'crohme_standard.png', invert=False)
        crohme_inverted = preprocess_and_save(crohme_img, 'crohme_inverted.png', invert=True)
        
        print(f'CROHME images saved to {crohme_standard} and {crohme_inverted}')
    
    # Process Supernote handwritten image
    supernote_img = '/app/assets/20250422_073317_Page_1.png'
    if os.path.exists(supernote_img):
        print(f'Processing Supernote image: {os.path.basename(supernote_img)}')
        
        # Process with and without inversion
        supernote_standard = preprocess_and_save(supernote_img, 'supernote_standard.png', invert=False)
        supernote_inverted = preprocess_and_save(supernote_img, 'supernote_inverted.png', invert=True)
        
        print(f'Supernote images saved to {supernote_standard} and {supernote_inverted}')
    
    print('\\nPreprocessing comparison completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"