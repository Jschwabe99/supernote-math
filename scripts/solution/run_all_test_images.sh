#!/bin/bash
# Script to run all test images through the PosFormer model and print results

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create a Python script to process all images
echo "Processing all test images in assets directory..."
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
        
        # Run beam search
        start_time = time.time()
        with torch.no_grad():
            try:
                hypotheses = model.beam_search(
                    tensor,
                    mask,
                    beam_size=5,  # Use smaller beam size for faster processing
                    max_len=100,  # Use smaller max_len for faster processing
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
    
    # Get all test images
    assets_dir = '/app/assets'
    test_images = [f for f in os.listdir(assets_dir) if f.endswith('.png')]
    test_images.sort()  # Sort to process them in order
    
    print(f'\\nFound {len(test_images)} test images\\n')
    
    # Process all images and print results
    print('-' * 80)
    print(f'| {\"Image\":<30} | {\"LaTeX\":<30} | {\"Score\":<10} | {\"Time(s)\":<10} |')
    print('-' * 80)
    
    for image_file in test_images:
        image_path = os.path.join(assets_dir, image_file)
        latex, score, process_time = recognize_image(image_path)
        
        # Truncate long LaTeX for display
        latex_display = latex[:28] + '...' if len(latex) > 30 else latex
        
        print(f'| {image_file:<30} | {latex_display:<30} | {score:<10.4f} | {process_time:<10.2f} |')
        
        # Also print the full LaTeX
        print(f'\\nFull LaTeX for {image_file}:')
        print(latex)
        print('-' * 80)
    
    print('\\nAll images processed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"