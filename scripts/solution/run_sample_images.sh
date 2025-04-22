#!/bin/bash
# Script to run a few sample test images through the PosFormer model

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create a Python script to process a few sample images
echo "Processing sample test images..."
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
    print('Model loaded successfully')
    model = lit_model.model
    model.eval()  # Set to evaluation mode
    
    # Function to process a single image
    def recognize_image(image_path):
        print(f'Processing image: {image_path}')
        
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
        
        # Create attention mask (0=attend, 1=ignore)
        mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
        
        # First, just run the encoder for verification
        print('Running encoder...')
        with torch.no_grad():
            feature, enc_mask = model.encoder(tensor, mask)
            print(f'Encoder output shape: {feature.shape}')
        
        # Try running beam search with very limited parameters for faster processing
        print('Running beam search (limited version)...')
        start_time = time.time()
        with torch.no_grad():
            try:
                hypotheses = model.beam_search(
                    tensor,
                    mask,
                    beam_size=3,  # Very small beam size
                    max_len=50,   # Short max length
                    alpha=1.0,
                    early_stopping=True,
                    temperature=1.0
                )
                
                # Process results
                if hypotheses:
                    best_hyp = hypotheses[0]
                    latex = vocab.indices2label(best_hyp.seq)
                    score = float(best_hyp.score)
                    return latex, score, time.time() - start_time
                else:
                    return 'No results', 0.0, time.time() - start_time
            except Exception as e:
                return f'Error: {str(e)}', 0.0, time.time() - start_time
    
    # Sample just a few images to test
    sample_images = [
        '/app/assets/20250422_073317_Page_1.png',
        '/app/assets/Test_Data_Page_1.png'
    ]
    
    # Process samples and print results
    print('\\n--- Processing Sample Images ---')
    for image_path in sample_images:
        if os.path.exists(image_path):
            latex, score, process_time = recognize_image(image_path)
            
            print(f'\\nResults for {os.path.basename(image_path)}:')
            print(f'LaTeX: {latex}')
            print(f'Score: {score:.4f}')
            print(f'Processing time: {process_time:.2f}s')
            print('-' * 50)
        else:
            print(f'Image not found: {image_path}')
    
    print('\\nSample images processed!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"