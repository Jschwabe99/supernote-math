#!/bin/bash
# Test script for a single image with timeout

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Process just a single image with a timeout
echo "Processing a single image with timeout..."
timeout 30s docker run --rm \
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

# Add paths
sys.path.append('/posformer')
sys.path.append('/app')

# Import components
try:
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.datamodule import vocab
    from core.data.loaders import preprocess_image
    
    # Load model
    print('Loading model...')
    checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
    
    # Load model directly
    lit_model = LitPosFormer.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    print('Model loaded successfully')
    model = lit_model.model
    model.eval()  # Set to evaluation mode
    
    # Process a single image
    image_path = '/app/assets/20250422_073317_Page_1.png'
    print(f'Processing image: {image_path}')
    
    # Load and preprocess
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    print('Preprocessing...')
    processed = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=False,
        normalize=False
    )
    
    # Convert to tensor with correct dimensions
    processed_np = processed.squeeze()  # Remove extra dimensions
    tensor = torch.from_numpy(processed_np).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    print(f'Tensor shape: {tensor.shape}')
    
    # Create attention mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Run encoder forward pass only (faster)
    print('Running encoder only...')
    with torch.no_grad():
        encoder_output, _ = model.encoder(tensor, mask)
        print(f'Encoder output shape: {encoder_output.shape}')
    
    print('Success! PosFormer model is working correctly.')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"