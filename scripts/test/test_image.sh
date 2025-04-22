#!/bin/bash
# Simplified script to test one image with the PosFormer model

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create a Python script to process just one image
echo "Processing a test image with detailed encoder output..."
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
    
    # Process just one image
    image_path = '/app/assets/20250422_073317_Page_1.png'
    print(f'\\nProcessing image: {image_path}')
    
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    print(f'Original image shape: {img_array.shape}')
    
    # Preprocess with specific settings
    processed = preprocess_image(
        image=img_array,
        target_resolution=(256, 1024),
        invert=False,
        normalize=False
    )
    print(f'Preprocessed shape: {processed.shape}')
    
    # Fix tensor format for the model
    processed_np = processed.squeeze()  # Remove extra dimensions
    print(f'After squeeze: {processed_np.shape}')
    
    tensor = torch.from_numpy(processed_np).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    print(f'Final tensor shape: {tensor.shape}')
    
    # Create attention mask
    mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
    
    # Run encoder only
    print('\\nRunning encoder forward pass...')
    with torch.no_grad():
        feature, enc_mask = model.encoder(tensor, mask)
        print(f'Encoder output shape: {feature.shape}')
    
    # Generate a small text sequence - just to test
    print('\\nTesting decoder with simple token generation...')
    with torch.no_grad():
        # Initialize with start token
        tgt = torch.tensor([[vocab.BOS_IDX]], device='cpu')
        
        # Prepare features for decoder
        feature_copy = feature.clone()
        feature_batch = torch.cat((feature_copy, feature_copy), dim=0)  # [2b, t, d]
        mask_batch = torch.cat((enc_mask, enc_mask), dim=0)
        
        # Test basic token generation - no full beam search
        for i in range(5):  # Generate 5 tokens only
            # Add batch dimension to match feature
            tgt_batch = torch.cat((tgt, tgt), dim=0)
            
            # Run through decoder
            out, _ = model.decoder(feature_batch, mask_batch, tgt_batch)
            
            # Get next token prediction
            next_token = torch.argmax(out[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Add to output
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Print token
            token_text = vocab.idx2token.get(next_token.item(), '<unk>')
            print(f'Generated token {i+1}: {token_text}')
        
        # Convert to LaTeX
        latex = vocab.indices2label(tgt[0].tolist())
        print(f'\\nPartial LaTeX output: {latex}')
    
    print('\\nTest completed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"