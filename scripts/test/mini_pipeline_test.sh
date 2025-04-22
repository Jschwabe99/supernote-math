#!/bin/bash
# Minimal test script for the PosFormer pipeline

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run a minimal pipeline test
echo "Testing minimal pipeline with image..."
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

# Add paths
sys.path.append('/posformer')
sys.path.append('/app')

# Import components
try:
    from app.pipeline import PosFormerRecognizer
    from core.data.loaders import preprocess_image
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.datamodule import vocab
    
    # Load model
    print('Loading model...')
    checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
    
    # Make sure the checkpoint exists
    print(f'Checkpoint exists: {os.path.exists(checkpoint_path)}')
    
    # Load model directly
    lit_model = LitPosFormer.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    print('Model loaded successfully')
    
    # Load and preprocess image
    image_path = '/app/assets/20250422_073317_Page_1.png'
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Preprocess with specific settings
    print('Preprocessing image...')
    processed = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=False,
        normalize=False
    )
    print(f'Processed image shape: {processed.shape}')
    
    # Convert to tensor
    tensor = torch.from_numpy(processed).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    print(f'Input tensor shape: {tensor.shape}')
    
    # Get the model
    model = lit_model.model
    
    # Create attention mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Set to eval mode
    model.eval()
    
    # Just do forward pass of encoder to see if it works
    print('Running model encoder only...')
    with torch.no_grad():
        enc_output = model.encode(tensor, mask)
        print(f'Encoder output shape: {enc_output.shape}')
    
    print('\\nTest successful!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"