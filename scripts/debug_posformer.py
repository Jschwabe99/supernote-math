#!/usr/bin/env python3
"""
Debug script for PosFormer model loading and image processing.

This script provides detailed diagnostics for:
1. The Python environment and dependencies
2. PosFormer model architecture and checkpoint loading
3. Image preprocessing and model prediction issues

Usage:
    python debug_posformer.py --model /path/to/checkpoint.ckpt --image /path/to/image.png
"""

import os
import sys
import argparse
import logging
import traceback
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check the current Python environment and dependencies."""
    logger.info("=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Try to import PosFormer modules
    try:
        logger.info("Attempting to import PosFormer modules...")
        
        # Add parent directory to path
        parent_dir = Path(__file__).resolve().parent.parent
        posformer_dir = parent_dir.parent / "PosFormer-main"
        
        logger.info(f"Parent directory: {parent_dir}")
        logger.info(f"PosFormer directory: {posformer_dir}")
        
        # Check if PosFormer directory exists
        if not posformer_dir.exists():
            logger.error(f"PosFormer directory not found: {posformer_dir}")
            return False
        
        # Add to sys.path
        sys.path.append(str(parent_dir))
        sys.path.append(str(posformer_dir))
        
        # Try importing PosFormer modules
        from Pos_Former.lit_posformer import LitPosFormer
        logger.info("Successfully imported LitPosFormer")
        
        return True
        
    except Exception as e:
        logger.error(f"Error importing PosFormer modules: {e}")
        logger.error(traceback.format_exc())
        return False

def debug_model_loading(model_path):
    """Debug model loading issues."""
    logger.info("=== Model Loading Diagnostics ===")
    
    try:
        # Check if model file exists
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Loading model from: {model_path}")
        
        # Try importing PosFormer modules
        from Pos_Former.lit_posformer import LitPosFormer
        
        # Examine checkpoint content
        logger.info("Examining checkpoint structure...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Print top-level keys
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # If 'state_dict' exists, examine its structure
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info(f"State dict contains {len(state_dict)} keys")
            
            # Print first 10 keys as sample
            sample_keys = list(state_dict.keys())[:10]
            logger.info(f"Sample keys: {sample_keys}")
            
            # Look for encoder/decoder patterns
            encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
            decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
            
            logger.info(f"Found {len(encoder_keys)} encoder-related keys")
            logger.info(f"Found {len(decoder_keys)} decoder-related keys")
            
            # Check for potential renaming patterns
            if encoder_keys:
                logger.info(f"Sample encoder keys: {encoder_keys[:5]}")
            if decoder_keys:
                logger.info(f"Sample decoder keys: {decoder_keys[:5]}")
        
        # Try loading the model
        logger.info("Attempting to load model with LitPosFormer.load_from_checkpoint...")
        
        # Set safety loading for PyTorch 2.0+
        try:
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            import torch.serialization
            torch.serialization.add_safe_globals([ModelCheckpoint])
        except Exception as e:
            logger.warning(f"Could not add safe globals: {e}")
        
        # Load model, catching specific exceptions
        try:
            lit_model = LitPosFormer.load_from_checkpoint(
                str(model_path),
                map_location='cpu'
            )
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model type: {type(lit_model).__name__}")
            
            model = lit_model.model
            logger.info(f"PosFormer model type: {type(model).__name__}")
            
            # Print model structure
            logger.info("Model structure overview:")
            logger.info(f"Encoder: {type(model.encoder).__name__}")
            logger.info(f"Decoder: {type(model.decoder).__name__}")
            
            return model
            
        except KeyError as e:
            logger.error(f"KeyError during model loading: {e}")
            logger.error("This suggests a mismatch between the code and checkpoint structure")
            logger.error(traceback.format_exc())
            
        except RuntimeError as e:
            logger.error(f"RuntimeError during model loading: {e}")
            logger.error("This suggests a tensor shape mismatch or CUDA issues")
            logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
        
        return None
        
    except Exception as e:
        logger.error(f"Error in debug_model_loading: {e}")
        logger.error(traceback.format_exc())
        return None

def debug_image_processing(image_path, model=None):
    """Debug image preprocessing and model prediction."""
    logger.info("=== Image Processing Diagnostics ===")
    
    try:
        # Check if image file exists
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return
        
        logger.info(f"Loading image from: {image_path}")
        
        # Load the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        logger.info(f"Original image size: {image.size}")
        
        # Import preprocessing components
        try:
            from core.data.loaders import preprocess_image
            logger.info("Successfully imported preprocessing function")
            
            # Convert to numpy array
            image_array = np.array(image)
            logger.info(f"Image array shape: {image_array.shape}")
            
            # Preprocess the image for PosFormer
            logger.info("Preprocessing image...")
            processed = preprocess_image(
                image=image_array,
                target_resolution=(256, 1024),  # PosFormer expected resolution
                invert=False,
                normalize=False,
                crop_margin=0.05,
                min_margin_pixels=20
            )
            
            logger.info(f"Processed image shape: {processed.shape}")
            logger.info(f"Processed image dtype: {processed.dtype}")
            logger.info(f"Processed image min: {processed.min()}, max: {processed.max()}")
            
            # If model is available, try running inference
            if model is not None:
                logger.info("Preparing image tensor for model inference...")
                
                # Convert to tensor [B, 1, H, W]
                tensor = torch.from_numpy(processed).float() / 255.0  # Normalize to [0, 1]
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch and channel dimensions
                logger.info(f"Input tensor shape: {tensor.shape}")
                
                # Create attention mask (all ones for attend everywhere)
                batch_size, channels, height, width = tensor.shape
                mask = torch.zeros((batch_size, height, width), dtype=torch.bool)
                
                # Set model to eval mode
                model.eval()
                
                # Try running beam search
                try:
                    logger.info("Running beam search inference...")
                    
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
                    
                    # Get the best hypothesis
                    if hypotheses:
                        from Pos_Former.datamodule import vocab
                        best_hyp = hypotheses[0]  # Sorted by score
                        latex = vocab.indices2label(best_hyp.seq)
                        confidence = float(best_hyp.score)
                        
                        logger.info(f"Recognition successful!")
                        logger.info(f"Recognized LaTeX: {latex}")
                        logger.info(f"Confidence: {confidence}")
                    else:
                        logger.warning("No hypotheses returned from beam search")
                        
                except Exception as e:
                    logger.error(f"Error during model inference: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Try encoder-only forward pass for simpler diagnostics
                    try:
                        logger.info("Trying encoder-only forward pass...")
                        enc_output = model.encode(tensor, mask)
                        logger.info(f"Encoder output shape: {enc_output.shape}")
                    except Exception as enc_e:
                        logger.error(f"Error in encoder forward pass: {enc_e}")
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"Error in debug_image_processing: {e}")
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Debug PosFormer model loading and inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    args = parser.parse_args()
    
    logger.info("===== Starting PosFormer Diagnostic Script =====")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        return
    
    # Debug model loading
    model = debug_model_loading(args.model)
    
    # Debug image processing
    debug_image_processing(args.image, model)
    
    logger.info("===== Diagnostics Complete =====")

if __name__ == "__main__":
    main()