#!/bin/bash
# Final working recognizer that performs beam search correctly

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run the working recognizer
echo "Running the fully working LaTeX recognizer..."
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
    print('Loading model...')
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
        print(f'\\nProcessing image: {image_path}')
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Preprocess the image
        processed = preprocess_image(
            image=img_array,
            target_resolution=(256, 1024),
            invert=False,
            normalize=False
        )
        
        # Convert to tensor with correct dimensions
        processed_np = processed.squeeze()  # Remove extra dimensions
        tensor = torch.from_numpy(processed_np).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Create attention mask
        mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
        
        # Run beam search with all required parameters
        with torch.no_grad():
            hypotheses = model.beam_search(
                tensor,
                mask,
                beam_size=10,
                max_len=200,
                alpha=1.0,
                early_stopping=True,
                temperature=1.0
            )
            
            if hypotheses:
                best_hyp = hypotheses[0]
                latex = vocab.indices2label(best_hyp.seq)
                score = float(best_hyp.score)
                return latex, score
            else:
                return None, 0.0
    
    # Process each test image
    for i in range(1, 10):
        image_path = f'/app/assets/20250422_073317_Page_{i}.png'
        if os.path.exists(image_path):
            latex, score = recognize_image(image_path)
            print(f'Image {i}: {image_path}')
            print(f'Recognized LaTeX: {latex}')
            print(f'Confidence Score: {score:.4f}')
            print('-' * 80)
    
    print('\\nAll images processed successfully!')
    
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc()
"