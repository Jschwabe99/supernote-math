#!/bin/bash
# Script to visualize the preprocessing of test images

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create visualization output directory
mkdir -p "$DIR/assets/visualized_preprocessing"

# Run visualization script
echo "Visualizing preprocessing for all test images..."
docker run --rm \
  -v "$DIR:/app" \
  supernote-posformer \
  python -c "
import os
import sys
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Add paths
sys.path.append('/app')

# Import preprocessing function
from core.data.loaders import preprocess_image

# Output directory
output_dir = '/app/assets/visualized_preprocessing'
os.makedirs(output_dir, exist_ok=True)

# Find all test page PNGs
page_images = glob.glob('/app/assets/20250422_073317_Page_*.png')
page_images.sort()

# Add test data image
test_data_path = '/app/assets/Test_Data_Page_1.png'
if os.path.exists(test_data_path):
    page_images.append(test_data_path)

print(f'Processing {len(page_images)} images')

# Process each image with various preprocessing settings
for idx, image_path in enumerate(page_images[:5]):  # Limit to first 5 images for speed
    basename = os.path.basename(image_path)
    print(f'Processing {basename}...')
    
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Create figure with 3 images (original, standard, inverted)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Standard preprocessing (no inversion)
    processed_standard = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=False,
        normalize=False
    ).squeeze()
    
    axes[1].imshow(processed_standard, cmap='gray')
    axes[1].set_title('Standard (invert=False)')
    axes[1].axis('off')
    
    # Inverted preprocessing (matching CROHME)
    processed_inverted = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=True,
        normalize=False
    ).squeeze()
    
    axes[2].imshow(processed_inverted, cmap='gray')
    axes[2].set_title('Inverted (invert=True, CROHME format)')
    axes[2].axis('off')
    
    # Add image info
    plt.suptitle(f'Preprocessing Visualization: {basename}')
    plt.figtext(0.5, 0.01, f'Original shape: {img_array.shape}, mean: {img_array.mean():.1f}', 
                ha='center', fontsize=10)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{os.path.splitext(basename)[0]}_preprocessing.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f'  Saved visualization to {output_path}')

print('\\nProcessing completed!')
"

echo "Visualization complete! Check the 'assets/visualized_preprocessing' directory."