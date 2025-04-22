#!/bin/bash
# Test script to examine PosFormer model structure

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run test to examine model structure
echo "Examining PosFormer model structure..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  supernote-posformer \
  python -c "
import os
import sys
import inspect
import torch

# Add paths
sys.path.append('/posformer')
sys.path.append('/app')

# Import components
try:
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.model.posformer import PosFormer
    
    # Load model
    print('Loading model...')
    checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
    
    # Load model directly
    lit_model = LitPosFormer.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    print('Model loaded successfully')
    
    # Get the model
    model = lit_model.model
    print(f'Model type: {type(model).__name__}')
    
    # Print model structure 
    print('\\nModel components:')
    for name, module in model.named_children():
        print(f'- {name}: {type(module).__name__}')
        
    # Print encoder and decoder methods
    print('\\nEncoder methods:')
    encoder_methods = [m for m in dir(model.encoder) if not m.startswith('_') and callable(getattr(model.encoder, m))]
    for method in encoder_methods[:10]:  # Show first 10 methods
        print(f'- {method}')
        
    print('\\nDecoder methods:')
    decoder_methods = [m for m in dir(model.decoder) if not m.startswith('_') and callable(getattr(model.decoder, m))]
    for method in decoder_methods[:10]:  # Show first 10 methods
        print(f'- {method}')
    
    # Print model methods
    print('\\nPosFormer model methods:')
    model_methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
    for method in model_methods:
        print(f'- {method}')
        
    # Try to find forward method
    print('\\nForward method signature:')
    if hasattr(model, 'forward'):
        sig = inspect.signature(model.forward)
        print(f'model.forward{sig}')
    
    # Try to find encoder forward method
    print('\\nEncoder forward method signature:')
    if hasattr(model.encoder, 'forward'):
        sig = inspect.signature(model.encoder.forward)
        print(f'model.encoder.forward{sig}')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"