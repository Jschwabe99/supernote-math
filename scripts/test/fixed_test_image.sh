#!/bin/bash
# Improved script to test one image with the PosFormer model

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create a Python script to process just one image
echo "Processing a test image with working decoder..."
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
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.datamodule import vocab
    from Pos_Former.utils.utils import Hypothesis
    from core.data.loaders import preprocess_image
    
    # Print vocabulary information
    print('Vocabulary information:')
    print(f'PAD_IDX: {vocab.PAD_IDX}')
    print(f'SOS_IDX: {vocab.SOS_IDX}')
    print(f'EOS_IDX: {vocab.EOS_IDX}')
    print(f'UNK_IDX: {vocab.UNK_IDX}')
    
    # Load model
    print('\\nLoading PosFormer model...')
    checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
    
    # Load model directly
    lit_model = LitPosFormer.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    print('Model loaded successfully!')
    model = lit_model.model
    model.eval()  # Set to evaluation mode
    
    # Process just one image
    image_path = '/app/assets/20250422_073317_Page_1.png'
    print(f'\\nProcessing image: {image_path}')
    
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Preprocess with specific settings
    processed = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=False,
        normalize=False
    )
    
    # Fix tensor format for the model
    processed_np = processed.squeeze()  # Remove extra dimensions
    tensor = torch.from_numpy(processed_np).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    print(f'Input tensor shape: {tensor.shape}')
    
    # Create attention mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Run encoder
    print('\\nRunning encoder forward pass...')
    with torch.no_grad():
        feature, enc_mask = model.encoder(tensor, mask)
        print(f'Encoder output shape: {feature.shape}')
    
    # Run beam search with restricted parameters to make it faster
    print('\\nRunning beam search with limited parameters...')
    with torch.no_grad():
        try:
            # Use very small beam search parameters for speed
            hypotheses = model.beam_search(
                tensor,
                mask,
                beam_size=3,      # Small beam size
                max_len=20,       # Short max length
                alpha=1.0,        # Standard length penalty
                early_stopping=True,
                temperature=1.0
            )
            
            # Process and print results
            if hypotheses:
                print('\\nBeam search results:')
                for i, hyp in enumerate(hypotheses[:3]):  # Show top 3 results
                    print(f'Hypothesis {i+1}:')
                    seq = hyp.seq
                    score = float(hyp.score)
                    
                    # Convert to tokens and LaTeX
                    tokens = [vocab.idx2token.get(idx, '<unk>') for idx in seq]
                    latex = vocab.indices2label(seq)
                    
                    print(f'  Score: {score:.4f}')
                    print(f'  Tokens: {tokens}')
                    print(f'  LaTeX: {latex}')
            else:
                print('No hypotheses generated')
        except Exception as e:
            print(f'Beam search error: {str(e)}')
            import traceback
            traceback.print_exc()
    
    print('\\nTest completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"