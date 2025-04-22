#!/bin/bash
# Test script for PosFormer using CROHME test image

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run test on CROHME dataset image
echo "Testing PosFormer on CROHME test image..."
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
    
    # Find CROHME test images
    print('Searching for CROHME test images...')
    crohme_images = glob.glob('/app/assets/test_output_*/CROHME_hf_*_comparison.png')
    
    if crohme_images:
        # Use the first CROHME image found
        crohme_img = crohme_images[0]
        print(f'Found CROHME test image: {os.path.basename(crohme_img)}')
        
        # Load and preprocess image
        img = Image.open(crohme_img).convert('L')
        img_array = np.array(img)
        
        # Show image info
        print(f'Image shape: {img_array.shape}')
        print(f'Value range: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.1f}')
        
        # For CROHME dataset, we need to determine if inversion is needed
        needs_inversion = img_array.mean() > 128  # If mean > 128, likely black-on-white
        
        print(f'Using invert={needs_inversion} based on image content')
        
        # Preprocess appropriately
        processed = preprocess_image(
            image=img_array,
            target_resolution=(256, 1024),
            invert=needs_inversion,
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
        
        # Run beam search with moderate parameters
        print('Running beam search...')
        start_time = time.time()
        
        with torch.no_grad():
            try:
                hypotheses = model.beam_search(
                    tensor,
                    mask,
                    beam_size=8,
                    max_len=50,
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
        
    else:
        print('No CROHME dataset images found in assets directory')
    
    print('\\nTest completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"