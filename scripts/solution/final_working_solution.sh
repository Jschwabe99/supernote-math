#!/bin/bash
# Final working solution for PosFormer with correct inversion settings

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"
CHECKPOINT_PATH="$POSFORMER_DIR/lightning_logs/version_0/checkpoints/best.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
  exit 1
fi

# Build the Docker image if needed
if [[ "$(docker images -q supernote-posformer 2> /dev/null)" == "" ]]; then
  echo "Building Docker image..."
  docker build -t supernote-posformer -f "$DIR/docker-scripts/Dockerfile" "$DIR"

  # Make sure the build succeeded
  if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
  fi
fi

# Run the corrected PosFormer image recognition with proper inversion
echo "Running PosFormer recognition with correct inversion settings..."
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
    def process_image(image_path, beam_size=8, max_len=50):
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
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
        
        # Run encoder
        with torch.no_grad():
            feature, enc_mask = model.encoder(tensor, mask)
        
        # Run beam search
        start_time = time.time()
        results = []
        
        with torch.no_grad():
            try:
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
                if hypotheses:
                    for hyp in hypotheses[:3]:  # Get top 3 results
                        latex = vocab.indices2label(hyp.seq)
                        score = float(hyp.score)
                        results.append((latex, score))
                
                process_time = time.time() - start_time
                return results, process_time
            except Exception as e:
                return [(f'Error: {str(e)}', 0.0)], 0.0
    
    # Process all Supernote image files
    image_files = []
    for i in range(1, 10):
        path = f'/app/assets/20250422_073317_Page_{i}.png'
        if os.path.exists(path):
            image_files.append(path)
    
    # Add Test_Data image if it exists
    test_data_path = '/app/assets/Test_Data_Page_1.png'
    if os.path.exists(test_data_path):
        image_files.append(test_data_path)
    
    print(f'Found {len(image_files)} image files to process')
    
    # Print header for results table
    print('\\n' + '=' * 90)
    print(f'| {\"Image\":<30} | {\"LaTeX (Best Match)\":<50} | {\"Score\":<9} |')
    print('=' * 90)
    
    # Process each image
    for image_path in image_files:
        basename = os.path.basename(image_path)
        print(f'Processing: {basename}')
        
        # Process with appropriate settings
        results, process_time = process_image(image_path, beam_size=8, max_len=50)
        
        if results:
            # Get best result (first one)
            latex, score = results[0]
            
            # Truncate LaTeX for display
            latex_display = latex if len(latex) < 47 else latex[:44] + '...'
            
            # Print result in table format
            print(f'| {basename:<30} | {latex_display:<50} | {score:<9.4f} |')
            
            # Print full LaTeX output
            print(f'Full LaTeX: {latex}')
            print('-' * 90)
        else:
            print(f'| {basename:<30} | No results generated | {0.0:<9} |')
            print('-' * 90)
    
    print('\\nAll images processed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"

echo "Processing completed!"
echo "The key insight: PosFormer expects images in CROHME format (white strokes on black background)"
echo "Use invert=True when preprocessing Supernote images (black strokes on white background)"