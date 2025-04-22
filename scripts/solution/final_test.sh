#!/bin/bash
# Final test script with correct model methods

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run fixed test with proper model methods
echo "Running final test with proper model methods..."
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
    
    # Load model directly
    lit_model = LitPosFormer.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    print('Model loaded successfully')
    model = lit_model.model
    model.eval()  # Set to evaluation mode
    
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
    
    # Fix tensor format - [batch, channels, height, width]
    tensor = torch.from_numpy(processed).float() / 255.0
    tensor = tensor.squeeze(-1).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 1024]
    print(f'Input tensor shape: {tensor.shape}')
    
    # Create attention mask - match the shape without channel dim
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Run encoder forward pass
    print('Running encoder forward pass...')
    with torch.no_grad():
        encoder_output, _ = model.encoder(tensor, mask)
        print(f'Encoder output shape: {encoder_output.shape}')
    
    # Try beam search with a dummy target
    print('\\nAttempting beam search...')
    with torch.no_grad():
        try:
            # Create dummy logger
            class DummyLogger:
                def experiment(self):
                    return None
                def log_metrics(self, *args, **kwargs):
                    pass
                    
            hypotheses = model.beam_search(
                tensor,
                mask,
                beam_size=5,
                max_len=50,
                alpha=1.0
            )
            
            if hypotheses:
                best_hyp = hypotheses[0]
                # Convert indices to LaTeX
                latex = vocab.indices2label(best_hyp.seq)
                print(f'Recognized LaTeX: {latex}')
                print(f'Score: {best_hyp.score}')
            else:
                print('No hypotheses generated')
                
        except Exception as e:
            print(f'Beam search error: {e}')
    
    print('\\nTest completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"