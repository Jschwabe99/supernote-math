"""Main application for math recognition and solving."""

import typer
import logging
import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import MathRecognitionPipeline
from utils.device_utils import configure_metal

app = typer.Typer(help="Run math recognition and solving")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.command()
def process_image(
    image_path: Path = typer.Option(..., help="Path to input image"),
    output_dir: Path = typer.Option(None, help="Directory to save results"),
    detection_model: Path = typer.Option(None, help="Path to detection model"),
    recognition_model: Path = typer.Option(None, help="Path to recognition model"),
    stroke_library: Path = typer.Option(None, help="Path to stroke library"),
    use_metal: bool = typer.Option(True, help="Use Metal GPU acceleration if available")
):
    """Process an image to recognize and solve math expressions."""
    # Configure Metal GPU if requested and available
    if use_metal:
        configure_metal()
    
    logger.info(f"Processing image: {image_path}")
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create pipeline
    pipeline = MathRecognitionPipeline(
        detection_model_path=detection_model,
        recognition_model_path=recognition_model,
        stroke_library_path=stroke_library
    )
    
    # Load image
    # In a real implementation, this would use a proper image loading library
    # like OpenCV's cv2.imread() or PIL's Image.open()
    
    # Placeholder - just create a dummy image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Process image
    results = pipeline.process_image(image)
    
    # Print results
    for i, result in enumerate(results):
        logger.info(f"Region {i+1}:")
        logger.info(f"  LaTeX: {result['latex']}")
        if result['solution']['solved']:
            logger.info(f"  Solution: {result['solution']['result']}")
        else:
            logger.info(f"  Solution: Failed - {result['solution']['error']}")
    
    logger.info(f"Processing completed with {len(results)} regions detected")

@app.command()
def interactive_mode(
    detection_model: Path = typer.Option(None, help="Path to detection model"),
    recognition_model: Path = typer.Option(None, help="Path to recognition model"),
    stroke_library: Path = typer.Option(None, help="Path to stroke library")
):
    """Run in interactive mode for Supernote integration."""
    logger.info("Starting interactive mode")
    
    # Create pipeline
    pipeline = MathRecognitionPipeline(
        detection_model_path=detection_model,
        recognition_model_path=recognition_model,
        stroke_library_path=stroke_library
    )
    
    # Placeholder for interactive mode
    # This would be implemented with the actual interactive mode code
    # for integration with Supernote
    logger.info("Interactive mode not yet implemented")
    
    logger.info("Interactive mode exited")

if __name__ == "__main__":
    app()