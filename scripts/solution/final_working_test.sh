#!/bin/bash
# Final working test script for PosFormer

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run the test with proper vocabulary handling
echo "Running the final working test for PosFormer..."
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
    
    # Print basic vocabulary info
    print('Vocabulary information:')
    print(f'PAD_IDX: {vocab.PAD_IDX}')
    print(f'SOS_IDX: {vocab.SOS_IDX}')
    print(f'EOS_IDX: {vocab.EOS_IDX}')
    print(f'Vocabulary size: {len(vocab.token2idx)}')
    
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
    
    # Function to process an image
    def process_image(image_path):
        print(f'\\nProcessing image: {os.path.basename(image_path)}')
        
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
        
        # Run encoder
        with torch.no_grad():
            feature, enc_mask = model.encoder(tensor, mask)
            print(f'Encoder output shape: {feature.shape}')
        
        # Run beam search
        print('Running beam search...')
        start_time = time.time()
        result = None
        
        with torch.no_grad():
            try:
                hypotheses = model.beam_search(
                    tensor,
                    mask,
                    beam_size=3,      # Small beam
                    max_len=20,       # Short sequences
                    alpha=1.0,
                    early_stopping=True,
                    temperature=1.0
                )
                
                if hypotheses:
                    best_hyp = hypotheses[0]
                    latex = vocab.indices2label(best_hyp.seq)
                    score = float(best_hyp.score)
                    result = (latex, score)
                else:
                    result = ('No results', 0.0)
            except Exception as e:
                print(f'Beam search error: {str(e)}')
                result = (f'Error: {str(e)}', 0.0)
        
        process_time = time.time() - start_time
        return result, process_time
    
    # Test with multiple images
    test_images = [
        '/app/assets/20250422_073317_Page_1.png',
        '/app/assets/20250422_073317_Page_2.png',
        '/app/assets/20250422_073317_Page_3.png'
    ]
    
    # Process and print results
    print('\\n' + '=' * 60)
    print(f'| {\"Image\":<25} | {\"LaTeX Result\":<25} | {\"Time(s)\":<8} |')
    print('=' * 60)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            (latex, score), process_time = process_image(image_path)
            
            # Truncate LaTeX for table display
            latex_display = latex if len(latex) < 22 else latex[:19] + '...'
            
            print(f'| {os.path.basename(image_path):<25} | {latex_display:<25} | {process_time:<8.2f} |')
            print(f'Full LaTeX: {latex}')
            print(f'Score: {score:.4f}')
            print('-' * 60)
        else:
            print(f'Image not found: {image_path}')
    
    print('\\nTests completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"