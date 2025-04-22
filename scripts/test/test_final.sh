#!/bin/bash
# Final test script with better beam search parameters

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run a final test with better parameters
echo "Running final test with better beam search parameters..."
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
import time

# Add paths
sys.path.append('/posformer')
sys.path.append('/app')

# Import components
try:
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.datamodule import vocab
    from core.data.loaders import preprocess_image
    
    # Load model
    print('Loading PosFormer model...')
    checkpoint_path = '/posformer/lightning_logs/version_0/checkpoints/best.ckpt'
    
    # Load model directly
    lit_model = LitPosFormer.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    print('Model loaded successfully!')
    model = lit_model.model
    model.eval()  # Set to evaluation mode
    
    # Process a single image with better parameters
    image_path = '/app/assets/20250422_073317_Page_1.png'
    print(f'\\nProcessing image: {os.path.basename(image_path)}')
    
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Show a quick summary of the image content
    print(f'Image shape: {img_array.shape}')
    print(f'Image min: {img_array.min()}, max: {img_array.max()}, mean: {img_array.mean():.1f}')
    
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
    
    # Create attention mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Run encoder
    print('Running encoder...')
    with torch.no_grad():
        feature, enc_mask = model.encoder(tensor, mask)
        print(f'Encoder output shape: {feature.shape}')
    
    # Run beam search with better parameters
    print('Running beam search with better parameters...')
    start_time = time.time()
    
    with torch.no_grad():
        try:
            hypotheses = model.beam_search(
                tensor,
                mask,
                beam_size=10,      # Larger beam size
                max_len=100,       # Longer sequences
                alpha=1.0,
                early_stopping=True,
                temperature=1.0
            )
            
            # Process results
            if hypotheses:
                print('\\nBeam search results:')
                for i, hyp in enumerate(hypotheses[:3]):  # Show top 3 results
                    latex = vocab.indices2label(hyp.seq)
                    score = float(hyp.score)
                    
                    print(f'Hypothesis {i+1}:')
                    print(f'  Score: {score:.4f}')
                    print(f'  LaTeX: {latex}')
            else:
                print('No results generated')
        except Exception as e:
            print(f'Beam search error: {str(e)}')
    
    process_time = time.time() - start_time
    print(f'Processing time: {process_time:.2f}s')
    
    print('\\nTest completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"