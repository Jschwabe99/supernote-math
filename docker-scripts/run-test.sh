#!/bin/bash
# Script to run a test with a specific image in the Docker container

set -e  # Exit on error

# Set up PYTHONPATH
export PYTHONPATH=/app:/posformer:$PYTHONPATH

# Run a simplified version of the test with the specified image
echo "=== Running simplified test with image: $1 ==="
cd /app
python -c "
import sys
import os
import torch
import numpy as np
from PIL import Image
import importlib.util

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('NumPy version:', np.__version__)
print('CUDA available:', torch.cuda.is_available())

print(f'Testing with image: \"{sys.argv[1]}\"')
if os.path.exists(sys.argv[1]):
    print('Image file exists')
    img = Image.open(sys.argv[1])
    print(f'Image size: {img.size}')
else:
    print('Image file does not exist')

# Import PosFormer modules
print('Attempting to import PosFormer modules...')
sys.path.append('/posformer')

# Try to import LitPosFormer
try:
    spec = importlib.util.find_spec('Pos_Former.lit_posformer')
    if spec is not None:
        print('Found LitPosFormer module')
        from Pos_Former.lit_posformer import LitPosFormer
        import torchvision.transforms as transforms
        print('Successfully imported LitPosFormer')
        
        # Try to load the model
        model_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
        print(f'Loading model from {model_path}')
        if os.path.exists(model_path):
            print('Model checkpoint exists')
            try:
                # Load the model
                lit_model = LitPosFormer.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
                print('Model loaded successfully!')
                
                model = lit_model.model
                print('Model architecture:', type(model).__name__)
                
                # Try to process the image with the model
                print('\\nAttempting to process the image with the model...')
                
                # Define image transformation
                transform = transforms.Compose([
                    transforms.Resize((128, 512)),
                    transforms.ToTensor(),
                ])
                
                # Transform the image
                img_transformed = transform(img).unsqueeze(0)
                print(f'Transformed image shape: {img_transformed.shape}')
                
                # Set model to evaluation mode
                model.eval()
                
                # Forward pass without gradient calculation
                with torch.no_grad():
                    try:
                        # Create img_mask (all ones, same batch size as img)
                        img_mask = torch.ones((1, 1, 128, 512), dtype=torch.bool)
                        
                        # Create dummy target (batch_size, max_len)
                        tgt = torch.zeros((1, 50), dtype=torch.long)
                        
                        # Create dummy logger
                        class DummyLogger:
                            def experiment(self):
                                return None
                            def log_metrics(self, *args, **kwargs):
                                pass
                        
                        logger = DummyLogger()
                        
                        # Try to run the model's encode method first (should require fewer arguments)
                        try:
                            enc_output = model.encode(img_transformed, img_mask)
                            print('Encoder forward pass completed successfully!')
                            print(f'Encoder output type: {type(enc_output)}')
                            if hasattr(enc_output, 'shape'):
                                print(f'Encoder output shape: {enc_output.shape}')
                        except Exception as e:
                            print(f'Error in encoder forward pass: {str(e)}')
                        
                        # Run full forward pass with required arguments
                        output = model(img_transformed, img_mask, tgt, logger)
                        print('Forward pass completed successfully!')
                        print(f'Output type: {type(output)}')
                        if hasattr(output, 'shape'):
                            print(f'Output shape: {output.shape}')
                    except Exception as e:
                        print(f'Error in forward pass: {str(e)}')
                
            except Exception as e:
                print(f'Error loading model: {str(e)}')
        else:
            print(f'Model checkpoint not found at {model_path}')
    else:
        print('Could not find LitPosFormer module')
except ImportError as e:
    print(f'Error importing modules: {str(e)}')
" "$1"

echo "=== Simplified test completed ==="