"""End-to-end pipeline for math recognition and solving with PosFormer."""

import numpy as np
import torch
import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from PIL import Image, ImageDraw

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add PosFormer to path
POSFORMER_PATH = Path(__file__).resolve().parent.parent.parent / 'PosFormer-main'
sys.path.append(str(POSFORMER_PATH))

# Import components after path setup
try:
    from Pos_Former.lit_posformer import LitPosFormer
    from Pos_Former.datamodule import vocab
    from Pos_Former.utils.utils import Hypothesis
except ImportError as e:
    logger.error(f"Error importing PosFormer components: {e}")
    logger.error(f"Make sure PosFormer is available at {POSFORMER_PATH}")
    raise

from core.solver.sympy_solver import SymPySolver
from core.render.handwriting import HandwritingRenderer, StrokeLibrary
from app.detection import MathRegionDetector
from core.data.loaders import preprocess_image


class Stroke:
    """Represents a single stroke from the Supernote device."""
    def __init__(self, points=None):
        self.points = points or []
    
    def add_point(self, x, y, pressure=1.0):
        """Add a point to the stroke."""
        self.points.append((x, y, pressure))
    
    def is_empty(self):
        """Check if the stroke is empty."""
        return len(self.points) == 0


class PosFormerRecognizer:
    """Math recognition using the PosFormer model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize the PosFormer recognizer.
        
        Args:
            model_path: Path to the pretrained model checkpoint
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Beam search parameters
        self.beam_config = {
            'beam_size': 10,
            'max_len': 200,
            'alpha': 1.0,
            'early_stopping': True,
            'temperature': 1.0,
        }
        
        # Load the model
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the pretrained PosFormer model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        try:
            # Load the model using LitPosFormer's load_from_checkpoint method
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                
                # Add safety loading for PyTorch 2.6+
                try:
                    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
                    import torch.serialization
                    torch.serialization.add_safe_globals([ModelCheckpoint])
                except Exception as e:
                    logger.warning(f"Could not add safe globals: {e}")
                
                # Load the model using the proper Lightning method
                lit_model = LitPosFormer.load_from_checkpoint(
                    model_path, 
                    map_location=self.device
                )
                
                # Get the actual model from the LitPosFormer wrapper
                self.model = lit_model.model
                
                # Move model to the appropriate device and set to eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model path does not exist: {model_path}")
                logger.warning("Model not loaded - results will be incorrect")
        
        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            raise
    
    def create_attention_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create attention mask for the model.
        
        Args:
            tensor: Input tensor [B, 1, H, W]
            
        Returns:
            Attention mask tensor [B, H, W]
        """
        # Create mask where 0 = attend, 1 = ignore
        batch_size, _, height, width = tensor.shape
        mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=self.device)
        return mask
    
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize handwritten math expression from image.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Dictionary containing recognition results
        """
        if self.model is None:
            return {'latex': '', 'error': 'Model not loaded'}
        
        try:
            start_time = time.time()
            
            # Preprocess image
            processed = preprocess_image(
                image=image,
                target_resolution=(256, 1024),  # PosFormer expects this resolution
                invert=False,  # No need to invert since we're drawing black on white
                normalize=False,  # Keep in range [0, 255] for model input
                crop_margin=0.05,
                min_margin_pixels=20
            )
            
            # Convert to tensor [B, 1, H, W]
            tensor = torch.from_numpy(processed).float() / 255.0  # Normalize to [0, 1]
            tensor = tensor.permute(0, 3, 1, 2).to(self.device)
            
            # Create attention mask
            mask = self.create_attention_mask(tensor)
            
            # Run inference with error handling
            try:
                with torch.no_grad():
                    hypotheses = self.model.beam_search(
                        tensor,
                        mask,
                        **self.beam_config
                    )
                
                # Get the best hypothesis
                if hypotheses:
                    best_hyp = hypotheses[0]  # Sorted by score
                    latex = vocab.indices2label(best_hyp.seq)
                    confidence = float(best_hyp.score)
                else:
                    latex = ""
                    confidence = 0.0
            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                latex = "Error during recognition"
                confidence = 0.0
            
            inference_time = time.time() - start_time
            
            return {
                'latex': latex,
                'confidence': confidence,
                'inference_time': inference_time
            }
        
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                'latex': '',
                'confidence': 0.0,
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def strokes_to_image(self, strokes: List[Stroke], canvas_width=1024, canvas_height=256, 
                         line_width=2, background_color=(255, 255, 255), stroke_color=(0, 0, 0)) -> Image.Image:
        """Convert a list of strokes to a PIL Image.
        
        Args:
            strokes: List of Stroke objects
            canvas_width: Width of the canvas
            canvas_height: Height of the canvas
            line_width: Width of the stroke lines
            background_color: Background color (default: white)
            stroke_color: Stroke color (default: black)
            
        Returns:
            PIL Image with rendered strokes
        """
        # Create a blank canvas
        image = Image.new('RGB', (canvas_width, canvas_height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Draw each stroke
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue  # Skip strokes with less than 2 points
            
            # Draw lines between consecutive points
            for i in range(len(stroke.points) - 1):
                x1, y1, p1 = stroke.points[i]
                x2, y2, p2 = stroke.points[i + 1]
                
                # Adjust line width based on pressure if needed
                width = int(line_width * max(p1, p2)) if max(p1, p2) > 0 else line_width
                
                # Draw the line
                draw.line([(x1, y1), (x2, y2)], fill=stroke_color, width=width)
        
        # Convert to grayscale
        image = image.convert('L')
        return image
    
    def recognize_strokes(self, strokes: List[Stroke]) -> Dict[str, Any]:
        """Recognize handwritten math expression from strokes.
        
        Args:
            strokes: List of Stroke objects containing the handwritten input
            
        Returns:
            Dictionary containing recognition results
        """
        # Convert strokes to image
        image = self.strokes_to_image(strokes)
        
        # Recognize from image
        image_array = np.array(image)
        return self.recognize(image_array)


class MathRecognitionPipeline:
    """End-to-end pipeline for math recognition, solving, and rendering."""
    
    def __init__(self, 
                 posformer_model_path: Optional[str] = None,
                 detection_model_path: Optional[Path] = None,
                 stroke_library_path: Optional[Path] = None):
        """Initialize pipeline with models.
        
        Args:
            posformer_model_path: Path to saved PosFormer model
            detection_model_path: Path to saved detection model
            stroke_library_path: Path to stroke library
        """
        # Find default model if not specified
        if posformer_model_path is None:
            # Look for checkpoint files in default location
            default_path = POSFORMER_PATH / 'lightning_logs/version_0/checkpoints'
            checkpoint_files = list(default_path.glob('*.ckpt'))
            
            if checkpoint_files:
                posformer_model_path = str(checkpoint_files[0])
                logger.info(f"Using default PosFormer model: {posformer_model_path}")
        
        # Initialize components with error handling
        try:
            self.detector = MathRegionDetector(detection_model_path)
        except Exception as e:
            logger.warning(f"Failed to initialize detector: {e}")
            self.detector = MathRegionDetector()
            
        try:
            self.recognizer = PosFormerRecognizer(posformer_model_path)
        except Exception as e:
            logger.warning(f"Failed to initialize recognizer: {e}")
            self.recognizer = PosFormerRecognizer()
            
        self.solver = SymPySolver()
        
        # Initialize rendering components if available
        self.renderer = None
        if stroke_library_path and stroke_library_path.exists():
            try:
                self.stroke_library = StrokeLibrary(stroke_library_path)
                self.renderer = HandwritingRenderer(self.stroke_library)
            except Exception as e:
                logger.warning(f"Failed to initialize renderer: {e}")
    
    def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Process document image to recognize and solve math expressions.
        
        Args:
            image: Input document image as NumPy array
            
        Returns:
            List of results, one per detected region
        """
        # Detect math regions
        regions = self.detector.detect_regions(image)
        
        # Extract region images
        region_images = self.detector.extract_regions(image, regions)
        
        # Process each region
        results = []
        for i, region_img in enumerate(region_images):
            # Run recognition on region
            recognition_result = self.recognizer.recognize(region_img)
            latex = recognition_result['latex']
            
            # Solve expression
            solution = self.solver.solve_latex(latex)
            
            # Render result if renderer is available
            rendered_result = None
            if self.renderer and solution['solved']:
                result_latex = solution['result_latex']
                rendered_result = self.renderer.render_latex(result_latex)
            
            # Add to results
            results.append({
                'region': regions[i],
                'latex': latex,
                'confidence': recognition_result.get('confidence', 0.0),
                'inference_time': recognition_result.get('inference_time', 0.0),
                'solution': solution,
                'rendered_result': rendered_result
            })
        
        return results
    
    def process_strokes(self, strokes: List[Stroke]) -> Dict[str, Any]:
        """Process handwritten strokes to recognize and solve math expressions.
        
        Args:
            strokes: List of Stroke objects containing handwritten input
            
        Returns:
            Dictionary containing recognition and solution results
        """
        # Recognize expression from strokes
        recognition_result = self.recognizer.recognize_strokes(strokes)
        latex = recognition_result['latex']
        
        # Solve expression
        solution = self.solver.solve_latex(latex)
        
        # Render result if renderer is available
        rendered_result = None
        if self.renderer and solution['solved']:
            result_latex = solution['result_latex']
            rendered_result = self.renderer.render_latex(result_latex)
        
        # Build complete result
        result = {
            'latex': latex,
            'confidence': recognition_result.get('confidence', 0.0),
            'inference_time': recognition_result.get('inference_time', 0.0),
            'solution': solution,
            'rendered_result': rendered_result
        }
        
        return result
    
    def extract_strokes_from_json(self, json_data: Dict[str, Any]) -> List[Stroke]:
        """Extract strokes from JSON data provided by the Supernote API.
        
        Args:
            json_data: JSON data containing stroke information
            
        Returns:
            List of Stroke objects
        """
        strokes = []
        
        try:
            # This is a placeholder - adjust according to the actual JSON format
            # from the Supernote device API
            for stroke_data in json_data.get('strokes', []):
                stroke = Stroke()
                
                for point in stroke_data.get('points', []):
                    x = point.get('x', 0)
                    y = point.get('y', 0)
                    pressure = point.get('pressure', 1.0)
                    stroke.add_point(x, y, pressure)
                
                if not stroke.is_empty():
                    strokes.append(stroke)
        
        except Exception as e:
            logger.error(f"Failed to extract strokes from JSON: {e}")
        
        return strokes
    
    def highlight_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """Highlight detected math regions in the image.
        
        Args:
            image: Input image as NumPy array
            regions: List of detected regions
            
        Returns:
            Image with highlighted regions
        """
        # Create a copy of the image
        highlighted = image.copy()
        
        # Draw rectangles around regions
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Draw rectangle
            # In a real implementation, this would use OpenCV's cv2.rectangle()
            highlighted[y1:y1+3, x1:x2] = [0, 255, 0]  # Top line
            highlighted[y2-3:y2, x1:x2] = [0, 255, 0]  # Bottom line
            highlighted[y1:y2, x1:x1+3] = [0, 255, 0]  # Left line
            highlighted[y1:y2, x2-3:x2] = [0, 255, 0]  # Right line
        
        return highlighted