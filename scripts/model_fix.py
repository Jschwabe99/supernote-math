#!/usr/bin/env python3
"""
Fix for PosFormer model loading issues.

This script modifies the model architecture to match the checkpoint structure.
It should be run after the Docker container is built but before running inference.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_model_architecture(checkpoint_path):
    """
    Fix model architecture to match checkpoint structure.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Fixing model architecture to match checkpoint at {checkpoint_path}")
        
        # Add parent directory to path
        parent_dir = Path(__file__).resolve().parent.parent
        posformer_dir = parent_dir.parent / "PosFormer-main"
        
        # Add to sys.path
        sys.path.append(str(parent_dir))
        sys.path.append(str(posformer_dir))
        
        # Import necessary modules
        from Pos_Former.lit_posformer import LitPosFormer
        from Pos_Former.model.posformer import PosFormer
        from Pos_Former.model.encoder import Encoder
        from Pos_Former.model.decoder import Decoder
        
        # Load checkpoint to examine its structure
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' not in checkpoint:
            logger.error("Checkpoint does not contain 'state_dict'")
            return False
        
        state_dict = checkpoint['state_dict']
        
        # Extract important information about the checkpoint
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
        
        # Find feature projection size
        feature_proj_keys = [k for k in state_dict.keys() if 'feature_proj' in k]
        
        if not feature_proj_keys:
            logger.error("No feature projection keys found in checkpoint")
            return False
        
        logger.info(f"Found feature projection keys: {feature_proj_keys}")
        
        # Determine feature projection dimensions
        for key in feature_proj_keys:
            if '.weight' in key:
                tensor = state_dict[key]
                logger.info(f"Feature projection weight shape: {tensor.shape}")
                
                # Patch the model to use this dimension
                output_dim, input_dim = tensor.shape[:2]
                logger.info(f"Output dimension: {output_dim}, Input dimension: {input_dim}")
                
                # Create a patched model class with the right dimensions
                class PatchedEncoder(Encoder):
                    def __init__(self, *args, **kwargs):
                        # Force the input_dim to match checkpoint
                        super().__init__(*args, **kwargs)
                        # Replace feature_proj layer with correct dimensions
                        self.feature_proj = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1)
                
                # Apply the patch by monkey-patching
                import Pos_Former.model.encoder
                Pos_Former.model.encoder.Encoder = PatchedEncoder
                
                logger.info("Successfully patched Encoder class to match checkpoint dimensions")
                return True
        
        logger.error("Could not find feature projection weight tensor in checkpoint")
        return False
                
    except Exception as e:
        logger.error(f"Error fixing model architecture: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Fix PosFormer model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()
    
    # Fix model architecture
    success = fix_model_architecture(args.checkpoint)
    
    if success:
        logger.info("Model architecture successfully fixed")
    else:
        logger.error("Failed to fix model architecture")
        sys.exit(1)

if __name__ == "__main__":
    main()