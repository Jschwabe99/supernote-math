#!/bin/bash
# Script to test all test pages with proper inversion for CROHME format

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"
CHECKPOINT_PATH="$POSFORMER_DIR/lightning_logs/version_0/checkpoints/best.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
  exit 1
fi

# Run test on all page PNG files with inversion
echo "Testing all page PNGs with CROHME-style inversion..."
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
import glob

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
    
    # Function to process an image
    def process_image(image_path, beam_size=5, max_len=30):
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Save original image statistics
        img_stats = {
            'shape': img_array.shape,
            'min': img_array.min(),
            'max': img_array.max(),
            'mean': img_array.mean()
        }
        
        # Preprocess WITH inversion to match CROHME format
        processed = preprocess_image(
            image=img_array,
            target_resolution=(256, 1024),
            invert=True,  # Use inversion to match CROHME's white-on-black format
            normalize=False
        )
        
        # Fix tensor format for the model
        processed_np = processed.squeeze()  # Remove extra dimensions
        tensor = torch.from_numpy(processed_np).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Create attention mask
        mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
        
        # Run inference
        start_time = time.time()
        try:
            # Run encoder
            with torch.no_grad():
                feature, enc_mask = model.encoder(tensor, mask)
            
            # Run beam search
            with torch.no_grad():
                hypotheses = model.beam_search(
                    tensor,
                    mask,
                    beam_size=beam_size,
                    max_len=max_len,
                    alpha=1.0,
                    early_stopping=True,
                    temperature=1.0
                )
                
                # Process results
                results = []
                if hypotheses:
                    for i, hyp in enumerate(hypotheses[:3]):  # Get top 3 results
                        latex = vocab.indices2label(hyp.seq)
                        score = float(hyp.score)
                        results.append((latex, score))
                
                process_time = time.time() - start_time
                return results, process_time, img_stats
        except Exception as e:
            return [(f'Error: {str(e)}', 0.0)], 0.0, img_stats
    
    # Find all test page PNGs
    page_images = glob.glob('/app/assets/20250422_073317_Page_*.png')
    
    # Add Test_Data image if it exists
    test_data_path = '/app/assets/Test_Data_Page_1.png'
    if os.path.exists(test_data_path):
        page_images.append(test_data_path)
    
    # Sort images to process them in order
    page_images.sort()
    
    print(f'Found {len(page_images)} test page images')
    
    # Print header for results table
    print('\\n' + '=' * 100)
    print(f'| {\"Image\":<25} | {\"Image Stats\":<25} | {\"LaTeX (Best Match)\":<40} | {\"Score\":<7} |')
    print('=' * 100)
    
    # Process each image
    for image_path in page_images:
        basename = os.path.basename(image_path)
        print(f'Processing: {basename}')
        
        # Process with appropriate settings
        results, process_time, img_stats = process_image(image_path, beam_size=5, max_len=30)
        
        if results:
            # Get best result (first one)
            latex, score = results[0]
            
            # Truncate LaTeX for display
            latex_display = latex if len(latex) < 37 else latex[:34] + '...'
            
            # Format image stats
            stats_str = f'{img_stats[\"shape\"]}, mean={img_stats[\"mean\"]:.1f}'
            
            # Print result in table format
            print(f'| {basename:<25} | {stats_str:<25} | {latex_display:<40} | {score:<7.4f} |')
            
            # Print full LaTeX output
            print(f'Full LaTeX: {latex}')
            print('-' * 100)
        else:
            print(f'| {basename:<25} | No results generated | | |')
            print('-' * 100)
    
    print('\\nAll images processed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"

echo "Processing completed!"
echo "Key insight: All images were processed with invert=True to match CROHME format"