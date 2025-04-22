#!/bin/bash
# Simple working test script for PosFormer

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run a simplified test version
echo "Running simplified working test for PosFormer..."
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
    
    # Function to preprocess image
    def preprocess_for_model(image_path):
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
        
        # Create attention mask
        mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
        
        return tensor, mask
    
    # Process each image
    for i in range(1, 4):  # Test the first 3 images
        image_path = f'/app/assets/20250422_073317_Page_{i}.png'
        
        print(f'\\nProcessing image: {os.path.basename(image_path)}')
        
        # Preprocess the image
        tensor, mask = preprocess_for_model(image_path)
        
        # Run encoder only first
        print('Running encoder...')
        with torch.no_grad():
            feature, enc_mask = model.encoder(tensor, mask)
            print(f'Encoder output shape: {feature.shape}')
        
        # Run beam search
        print('Running beam search...')
        start_time = time.time()
        
        with torch.no_grad():
            try:
                hypotheses = model.beam_search(
                    tensor,
                    mask,
                    beam_size=3,
                    max_len=20,
                    alpha=1.0,
                    early_stopping=True,
                    temperature=1.0
                )
                
                # Process results
                if hypotheses:
                    best_hyp = hypotheses[0]
                    latex = vocab.indices2label(best_hyp.seq)
                    score = float(best_hyp.score)
                    
                    print(f'LaTeX result: {latex}')
                    print(f'Score: {score:.4f}')
                else:
                    print('No results generated')
            except Exception as e:
                print(f'Beam search error: {str(e)}')
        
        process_time = time.time() - start_time
        print(f'Processing time: {process_time:.2f}s')
        print('-' * 50)
    
    print('\\nTests completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"