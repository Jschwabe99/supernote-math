#!/bin/bash
# Test script focusing on model loading

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run a test focusing on model loading
echo "Testing model loading..."
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

# Check if checkpoint exists
checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
print(f'Checkpoint exists: {os.path.exists(checkpoint_path)}')

try:
    print('Loading checkpoint to examine structure...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('Checkpoint keys:', list(checkpoint.keys()))
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f'State dict keys count: {len(state_dict.keys())}')
        
        # Print sample keys
        sample_keys = list(state_dict.keys())[:10]
        print('Sample keys:', sample_keys)
        
        # Extract feature projection dimensions
        feature_proj_keys = [k for k in state_dict.keys() if 'feature_proj' in k]
        print('Feature projection keys:', feature_proj_keys)
        
        for key in feature_proj_keys:
            if '.weight' in key:
                tensor = state_dict[key]
                print(f'Feature projection weight shape: {tensor.shape}')
    
    # Try to import and load model
    print('\\nAttempting to import PosFormer models...')
    from Pos_Former.lit_posformer import LitPosFormer
    
    print('Loading model with LitPosFormer.load_from_checkpoint...')
    try:
        lit_model = LitPosFormer.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu'
        )
        print('Model loaded successfully!')
        print('Model type:', type(lit_model).__name__)
        
        # Print model structure
        model = lit_model.model
        print('PosFormer model type:', type(model).__name__)
        print('Encoder type:', type(model.encoder).__name__)
        print('Decoder type:', type(model.decoder).__name__)
        
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        import traceback
        traceback.print_exc()
        
        # Print the first few stack frames
        print('\\nDetailed error analysis:')
        import inspect
        tb = sys.exc_info()[2]
        while tb:
            frame = tb.tb_frame
            tb = tb.tb_next
            filename = frame.f_code.co_filename
            name = frame.f_code.co_name
            print(f'File \"{filename}\", line {frame.f_lineno}, in {name}')
            
            # If in encoder or decoder, print locals
            if 'encoder' in filename.lower() or 'decoder' in filename.lower():
                print('Local variables:')
                for key, value in frame.f_locals.items():
                    if isinstance(value, torch.Tensor):
                        print(f'  {key}: Tensor of shape {value.shape}')
                    elif hasattr(value, '__name__'):
                        print(f'  {key}: {value.__class__.__name__} ({value.__name__})')
                    else:
                        print(f'  {key}: {value.__class__.__name__}')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"