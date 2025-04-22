#!/bin/bash
# Simple test script to run just one diagnostic in the Docker container

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run just one test with a single image for debugging
echo "Running single diagnostic test..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer \
  python -c "
import os
import sys
import torch
import numpy as np
from PIL import Image

# Add paths
sys.path.append('/posformer')
sys.path.append('/app')

# Print environment info
print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('NumPy version:', np.__version__)
print('CUDA available:', torch.cuda.is_available())

# Check if PosFormer modules can be imported
try:
    print('Importing PosFormer modules...')
    from Pos_Former.lit_posformer import LitPosFormer
    print('Successfully imported LitPosFormer')
    
    # Check if checkpoint exists
    checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
    print(f'Checkpoint exists: {os.path.exists(checkpoint_path)}')
    
    # Load the test image
    image_path = '/app/assets/20250422_073317_Page_1.png'
    print(f'Image exists: {os.path.exists(image_path)}')
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        print(f'Image size: {img.size}, mode: {img.mode}')
        
        # Display image info
        img_array = np.array(img)
        print(f'Image array shape: {img_array.shape}, dtype: {img_array.dtype}')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"