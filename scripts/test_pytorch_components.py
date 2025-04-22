#!/usr/bin/env python3
"""
Test script to verify PyTorch components work properly.

This script:
1. Verifies that PyTorch can be imported
2. Tests if a basic PyTorch operation works
3. Tests if the MathRegionDetector can be initialized
4. Verifies torchvision functionality if available
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pytorch_import():
    """Test if PyTorch can be imported."""
    print("Testing PyTorch import...")
    
    try:
        import torch
        logger.info(f"PyTorch {torch.__version__} successfully imported")
        
        # Test a basic PyTorch operation
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = torch.matmul(tensor, tensor)
        logger.info(f"PyTorch operation result: {result}")
        
        # Check device availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA is not available, using CPU")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import PyTorch: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running PyTorch operation: {e}")
        return False

def test_torchvision_import():
    """Test if torchvision can be imported."""
    print("Testing torchvision import...")
    
    try:
        import torchvision
        logger.info(f"torchvision {torchvision.__version__} successfully imported")
        
        # Test a basic torchvision model
        try:
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)
            logger.info("Successfully created ResNet-18 model")
            
            # Test detection models if applicable
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                detection_model = fasterrcnn_resnet50_fpn(pretrained=False)
                logger.info("Successfully created Faster R-CNN model")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Detection models not available: {e}")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Model imports not available: {e}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import torchvision: {e}")
        return False
    except Exception as e:
        logger.error(f"Error with torchvision: {e}")
        return False
        
def test_pytorch_lightning_import():
    """Test if PyTorch Lightning can be imported."""
    print("Testing PyTorch Lightning import...")
    
    try:
        import pytorch_lightning as pl
        logger.info(f"PyTorch Lightning {pl.__version__} successfully imported")
        
        # Test a basic lightning module
        class LitModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                import torch.nn as nn
                self.layer = nn.Linear(10, 10)
                
            def forward(self, x):
                return self.layer(x)
        
        model = LitModel()
        logger.info("Successfully created Lightning model")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import PyTorch Lightning: {e}")
        return False
    except Exception as e:
        logger.error(f"Error with PyTorch Lightning: {e}")
        return False

def test_math_region_detector():
    """Test if the MathRegionDetector can be initialized."""
    print("Testing MathRegionDetector initialization...")
    
    try:
        from app.detection import MathRegionDetector
        
        detector = MathRegionDetector()
        logger.info("MathRegionDetector successfully initialized")
        
        # Test basic detection functionality with a dummy image
        try:
            # Create a dummy image
            img = np.zeros((100, 100), dtype=np.uint8)
            # Draw a rectangle
            img[30:70, 30:70] = 255
            
            # Run detection
            regions = detector.detect_regions(img)
            logger.info(f"Detection successful, found {len(regions)} regions")
            
            if regions:
                logger.info(f"Example region: {regions[0]}")
        except Exception as e:
            logger.error(f"Error testing detection: {e}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import MathRegionDetector: {e}")
        return False
    except Exception as e:
        logger.error(f"Error initializing MathRegionDetector: {e}")
        return False

def main():
    """Run all tests."""
    print("\n======== Testing PyTorch Components ========\n")
    
    # Run tests
    pt_success = test_pytorch_import()
    tv_success = test_torchvision_import()
    pl_success = test_pytorch_lightning_import()
    detector_success = test_math_region_detector()
    
    # Print summary
    print("\n======== Test Results ========\n")
    print(f"PyTorch import: {'✅ Success' if pt_success else '❌ Failed'}")
    print(f"torchvision import: {'✅ Success' if tv_success else '❌ Failed'}")
    print(f"PyTorch Lightning: {'✅ Success' if pl_success else '❌ Failed'}")
    print(f"MathRegionDetector: {'✅ Success' if detector_success else '❌ Failed'}")
    
    # Overall result
    all_success = pt_success and detector_success  # Consider these critical
    if all_success:
        print("\n✅ Critical components are working!")
        if not tv_success or not pl_success:
            print("Some non-critical components had issues but the system should still function.")
    else:
        print("\n❌ Some critical components failed.")
        if not pt_success:
            print("PyTorch is not available. This is required for the system to function.")
        if not detector_success:
            print("MathRegionDetector could not be initialized. Check the implementation.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())