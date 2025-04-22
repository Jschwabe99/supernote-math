# Supernote-Math Project Documentation

## Project Overview
The Supernote-Math project integrates the PosFormer AI model for handwritten mathematical expression recognition with the Supernote device. The system takes stroke information or images from the Supernote, processes them, and converts them to LaTeX representations.

## Directory Structure

```
supernote-math/
├── app/                # Main application components
│   ├── detection.py    # Mathematical region detection using OpenCV
│   ├── pipeline.py     # End-to-end math recognition pipeline
│   └── __init__.py
├── cli/                # Command-line interface tools
│   ├── benchmark.py    # Performance benchmarking
│   ├── convert.py      # Format conversion utilities
│   ├── run.py          # Main CLI runner
│   ├── train.py        # Training utilities
│   └── __init__.py
├── config.py           # Centralized project configuration
├── core/               # Core functionality modules
│   ├── data/           # Data handling and preprocessing
│   │   ├── augmentations.py   # Data augmentation methods
│   │   ├── loaders/           # Dataset loaders
│   │   └── __init__.py
│   ├── model/          # Model components
│   │   ├── recognition.py     # Recognition models
│   │   ├── tokenizer.py       # Tokenization utilities
│   │   └── __init__.py
│   ├── render/         # Rendering utilities
│   │   ├── handwriting.py     # Handwriting rendering
│   │   └── __init__.py
│   ├── solver/         # Math solving modules
│   │   ├── sympy_solver.py    # SymPy-based math solver
│   │   └── __init__.py
│   └── __init__.py
├── Docker/             # Docker configuration files
│   └── Dockerfile      # Container definition for Python 3.7
├── run_tests.sh        # Script to run tests with correct environment
├── scripts/            # Utility scripts
│   ├── analyze_dataset_sizes.py     # Dataset analysis
│   ├── explore_dataset.py           # Dataset exploration
│   ├── metal_check.py               # Metal GPU compatibility check
│   ├── process_all_datasets.py      # Dataset processing
│   ├── test_mock.py                 # Mock testing
│   ├── test_posformer_pipeline.py   # PosFormer integration tests
│   ├── test_preprocessing.py        # Preprocessing tests
│   ├── test_with_examples.py        # Example-based testing
│   └── visualize_preprocessed.py    # Visualization tools
├── setup.py            # Package setup script
├── setup_posformer_env.sh  # Environment setup script
├── tests/              # Test suite
│   └── __init__.py
└── utils/              # Utility functions
    ├── cli_utils.py    # CLI utilities
    ├── device_utils.py # Device-specific utilities
    ├── logging_utils.py# Logging utilities
    └── __init__.py
```

## Key Components

### 1. app/pipeline.py
The core pipeline that handles the end-to-end process of recognizing and solving mathematical expressions.

```python
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
```

### 2. app/detection.py
Module for detecting mathematical regions in handwritten notes.

```python
"""Detection of math regions in handwritten notes using OpenCV."""

import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathRegionDetector:
    """Detector for math regions in handwritten notes using OpenCV."""
    
    def __init__(self, model_path: Path = None):
        """Initialize detector.
        
        Args:
            model_path: Path to a detection model (optional, not used in this implementation)
        """
        # This implementation uses OpenCV for detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info("Using OpenCV for math region detection")
        
        if model_path:
            logger.info(f"Note: Model path {model_path} provided but not used in this implementation")
    
    def detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect math regions in a document image.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            List of detected regions, each a dict with 'bbox' and 'confidence' keys
        """
        height, width = image.shape[:2]
        
        try:
            import cv2
            return self._opencv_detect_regions(image)
        except ImportError:
            # Basic fallback if OpenCV is not available
            logger.warning("OpenCV not available, using basic fallback")
            regions = [
                {
                    'bbox': [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
                    'confidence': 0.5,  # Low confidence for fallback
                    'type': 'math'
                }
            ]
            return regions
    
    def _opencv_detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """OpenCV-based implementation for region detection.
        
        Args:
            image: Input image
            
        Returns:
            Detected regions
        """
        import cv2
        
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Invert image if it's white on black
        if np.mean(gray) < 127:
            gray = 255 - gray
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphology to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter based on size and aspect ratio
            min_area = 1000
            min_aspect_ratio = 0.2
            max_aspect_ratio = 5.0
            
            if (area > min_area and 
                aspect_ratio > min_aspect_ratio and 
                aspect_ratio < max_aspect_ratio):
                
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                regions.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.8,  # Medium confidence for OpenCV detection
                    'type': 'math'
                })
        
        # If no regions found, use the whole image
        if not regions:
            logger.info("No regions detected, using entire image")
            regions = [{
                'bbox': [0, 0, width, height],
                'confidence': 0.5,
                'type': 'math'
            }]
            
        return regions
    
    def extract_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract region images from the document image.
        
        Args:
            image: Input image as NumPy array
            regions: List of detected regions
            
        Returns:
            List of region images
        """
        region_images = []
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Ensure coordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Skip invalid regions
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid region coordinates: {x1}, {y1}, {x2}, {y2}")
                continue
                
            region_img = image[y1:y2, x1:x2]
            region_images.append(region_img)
        
        return region_images
```

### 3. core/solver/sympy_solver.py
Module for solving mathematical expressions using SymPy.

```python
"""SymPy-based solver for mathematical expressions."""

import re
import sympy
from sympy.parsing.latex import parse_latex
from typing import Optional, Union, Dict, Any, Tuple

from config import MAX_SOLVING_TIME


class LatexParser:
    """Parser for LaTeX expressions to SymPy format."""
    
    def __init__(self):
        """Initialize LaTeX parser."""
        pass
    
    def parse_to_sympy(self, latex_string: str) -> Optional[sympy.Expr]:
        """Parse LaTeX string to SymPy expression.
        
        Args:
            latex_string: LaTeX string to parse
            
        Returns:
            SymPy expression or None if parsing failed
        """
        try:
            # Remove unnecessary LaTeX formatting
            cleaned_latex = self._clean_latex(latex_string)
            # Parse LaTeX to SymPy expression
            expr = parse_latex(cleaned_latex)
            return expr
        except Exception as e:
            print(f"Error parsing LaTeX: {e}")
            return None
    
    def _clean_latex(self, latex_string: str) -> str:
        """Clean LaTeX string for parsing.
        
        Args:
            latex_string: Original LaTeX string
            
        Returns:
            Cleaned LaTeX string
        """
        # Remove display math delimiters
        latex_string = latex_string.replace('\\[', '').replace('\\]', '')
        latex_string = latex_string.replace('\\begin{equation}', '').replace('\\end{equation}', '')
        
        # Add other cleaning rules as needed
        return latex_string


class SymPySolver:
    """Solver for mathematical expressions using SymPy."""
    
    def __init__(self, timeout: float = MAX_SOLVING_TIME):
        """Initialize solver with timeout.
        
        Args:
            timeout: Maximum time in seconds to spend on solving
        """
        self.parser = LatexParser()
        self.timeout = timeout
    
    def solve_latex(self, latex_string: str) -> Dict[str, Any]:
        """Solve a LaTeX mathematical expression.
        
        This method detects the type of expression (equation, calculation, etc.)
        and applies the appropriate solving method.
        
        Args:
            latex_string: LaTeX string to solve
            
        Returns:
            Dictionary with solution information
        """
        result = {
            'input': latex_string,
            'parsed': False,
            'solved': False,
            'result': None,
            'result_latex': None,
            'error': None,
            'solution_type': None,
        }
        
        # Parse LaTeX to SymPy expression
        try:
            expr = self.parser.parse_to_sympy(latex_string)
            if expr is None:
                result['error'] = "Failed to parse LaTeX"
                return result
                
            result['parsed'] = True
            
            # Detect expression type and solve accordingly
            if '=' in latex_string:
                # Equation solving
                result['solution_type'] = 'equation'
                solution = self._solve_equation(expr)
                result['result'] = solution
                result['solved'] = True
            else:
                # Expression evaluation
                result['solution_type'] = 'evaluation'
                value = self._evaluate_expression(expr)
                result['result'] = value
                result['solved'] = True
            
            # Convert result back to LaTeX
            if result['result'] is not None:
                result['result_latex'] = sympy.latex(result['result'])
                
        except Exception as e:
            result['error'] = f"Error solving expression: {str(e)}"
        
        return result
    
    def _solve_equation(self, expr: sympy.Expr) -> Union[sympy.Expr, Dict]:
        """Solve an equation for unknown variables.
        
        Args:
            expr: SymPy equation expression
            
        Returns:
            Solution as SymPy expression or dictionary
        """
        # This is a placeholder implementation
        # In reality, would use sympy.solve() with proper equation handling
        if isinstance(expr, sympy.Equality):
            # Extract LHS and RHS
            lhs = expr.lhs
            rhs = expr.rhs
            
            # Find free symbols (variables)
            symbols = expr.free_symbols
            if len(symbols) == 0:
                # No variables, just check if equation is true
                return sympy.sympify(lhs == rhs)
            elif len(symbols) == 1:
                # One variable, solve for it
                symbol = list(symbols)[0]
                return sympy.solve(lhs - rhs, symbol)
            else:
                # Multiple variables, solve for all
                return {str(symbol): sympy.solve(lhs - rhs, symbol) for symbol in symbols}
        else:
            # Not an equality, can't solve as equation
            return expr
    
    def _evaluate_expression(self, expr: sympy.Expr) -> sympy.Expr:
        """Evaluate a mathematical expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            Evaluated expression
        """
        # Check if expression is numeric (no free variables)
        if len(expr.free_symbols) == 0:
            # Fully numeric, evaluate to float
            return float(expr.evalf())
        else:
            # Contains variables, simplify expression
            return expr.simplify()
    
    def format_result(self, result: Dict[str, Any], fmt: str = 'latex') -> str:
        """Format the solution result in the requested format.
        
        Args:
            result: Result dictionary from solve_latex
            fmt: Output format ('latex', 'text', 'mathml')
            
        Returns:
            Formatted result string
        """
        if not result['solved']:
            return f"Error: {result['error']}" if result['error'] else "Could not solve expression"
        
        if fmt == 'latex':
            return result['result_latex'] if result['result_latex'] else sympy.latex(result['result'])
        elif fmt == 'text':
            return str(result['result'])
        elif fmt == 'mathml':
            return sympy.printing.mathml(result['result'])
        else:
            return str(result['result'])


def detect_question_in_latex(latex_string: str) -> Tuple[str, bool]:
    """Detect if LaTeX string contains a question to solve.
    
    Args:
        latex_string: LaTeX string to analyze
        
    Returns:
        Tuple of (processed_latex, is_question)
    """
    # Check for question marks
    has_question_mark = '?' in latex_string
    
    # Check for equals signs with no right-hand side
    has_empty_rhs = bool(re.search(r'=\s*$', latex_string))
    
    # Replace question marks with appropriate SymPy variables
    processed_latex = latex_string.replace('?', 'x')
    
    return processed_latex, (has_question_mark or has_empty_rhs)
```

### 4. scripts/test_with_examples.py
Script for testing the PosFormer pipeline with example images.

```python
#!/usr/bin/env python3
"""
Test script for the PosFormer-based math recognition pipeline using real examples.

This script:
1. Loads test images with handwritten mathematical expressions
2. Processes individual expressions or segments a page into multiple expressions
3. Runs each expression through the PosFormer model
4. Displays the recognized LaTeX output and solutions
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import json
import time
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import MathRecognitionPipeline, PosFormerRecognizer

def segment_expressions(image, min_area=1000, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """
    Segment a page into individual math expressions using contour detection.
    
    Args:
        image: Input image array (grayscale)
        min_area: Minimum area of an expression region
        min_aspect_ratio: Minimum width/height ratio to be considered valid
        max_aspect_ratio: Maximum width/height ratio to be considered valid
        
    Returns:
        List of (x, y, w, h) bounding boxes for detected expressions
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Invert image if it's white on black
    if np.mean(gray) < 127:
        gray = 255 - gray
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Apply some morphology to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    expression_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        if (area > min_area and 
            aspect_ratio > min_aspect_ratio and 
            aspect_ratio < max_aspect_ratio):
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            
            expression_boxes.append((x, y, w, h))
    
    # Sort boxes from top to bottom
    expression_boxes.sort(key=lambda box: box[1])
    
    return expression_boxes

def process_page(image_path, model_path=None, show_result=True):
    """
    Process a page with multiple math expressions.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model checkpoint
        show_result: Whether to display the results
    """
    print(f"Processing page with math expressions: {image_path}")
    
    try:
        # Load the image
        image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
        
        # Segment expressions
        expression_boxes = segment_expressions(image)
        print(f"Detected {len(expression_boxes)} expressions")
        
        # Initialize the pipeline
        pipeline = MathRecognitionPipeline(posformer_model_path=model_path)
        
        # Process each expression
        results = []
        for i, (x, y, w, h) in enumerate(expression_boxes):
            # Extract expression region
            expr_image = image[y:y+h, x:x+w]
            
            # Process the expression
            start_time = time.time()
            result = pipeline.recognizer.recognize(expr_image)
            inference_time = time.time() - start_time
            
            # Add position info to result
            result['position'] = (x, y, w, h)
            result['index'] = i + 1
            
            # Try to solve the expression
            solution = pipeline.solver.solve_latex(result['latex'])
            result['solution'] = solution
            
            results.append(result)
            
            # Print results
            print(f"Expression {i+1}:")
            print(f"  Position: x={x}, y={y}, w={w}, h={h}")
            print(f"  Recognized: {result['latex']}")
            print(f"  Confidence: {result.get('confidence', 0):.4f}")
            print(f"  Inference time: {inference_time:.4f}s")
            
            if solution['solved']:
                print(f"  Solution: {solution['result_latex']}")
            else:
                print(f"  Could not solve: {solution.get('error', 'Unknown error')}")
            print()
        
        # Show the results
        if show_result and results:
            plt.figure(figsize=(12, 10))
            
            # Display original image with bounding boxes
            plt.imshow(image, cmap='gray')
            
            # Add bounding boxes and labels
            for result in results:
                x, y, w, h = result['position']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add label with recognition result
                label = f"{result['index']}: {result['latex']}"
                if result['solution']['solved']:
                    label += f" = {result['solution']['result_latex']}"
                plt.text(x, y-5, label, color='red', fontsize=8, 
                         bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title(f"Math Expressions Recognition")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return results
    
    except Exception as e:
        print(f"Error processing page: {e}")
        return []

def process_single_expression(image_path, model_path=None, show_result=True):
    """
    Process a single math expression image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model checkpoint
        show_result: Whether to display the results
    """
    print(f"Processing single math expression: {image_path}")
    
    try:
        # Load the image
        image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
        
        # Initialize the pipeline
        pipeline = MathRecognitionPipeline(posformer_model_path=model_path)
        
        # Process the expression
        start_time = time.time()
        result = pipeline.recognizer.recognize(image)
        inference_time = time.time() - start_time
        
        # Try to solve the expression
        solution = pipeline.solver.solve_latex(result['latex'])
        
        # Print results
        print(f"Recognized: {result['latex']}")
        print(f"Confidence: {result.get('confidence', 0):.4f}")
        print(f"Inference time: {inference_time:.4f}s")
        
        if solution['solved']:
            print(f"Solution: {solution['result_latex']}")
        else:
            print(f"Could not solve: {solution.get('error', 'Unknown error')}")
        
        # Show the result
        if show_result:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 1, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Input | Recognized: {result['latex']}")
            if solution['solved']:
                plt.xlabel(f"Solution: {solution['result_latex']}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return result
    
    except Exception as e:
        print(f"Error processing expression: {e}")
        return {'latex': '', 'error': str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test PosFormer pipeline with examples')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    parser.add_argument('--page', action='store_true', help='Process as a page with multiple expressions')
    parser.add_argument('--test-page', action='store_true', 
                      help='Use the default Test_Data_Page_1.png')
    args = parser.parse_args()
    
    # Choose image path
    image_path = args.image
    if args.test_page or (not image_path and not args.page):
        image_path = str(Path(__file__).resolve().parent.parent / 'assets' / 'Test_Data_Page_1.png')
        args.page = True
    
    if not image_path:
        print("Error: Please provide an image path with --image or use --test-page")
        return
    
    # Process the image
    if args.page:
        process_page(image_path, args.model, not args.no_show)
    else:
        process_single_expression(image_path, args.model, not args.no_show)

if __name__ == "__main__":
    main()
```

### 5. config.py
Centralized configuration settings for the project.

```python
"""
Centralized configuration for the Supernote Math Recognition project.
This module contains configuration classes for all components of the system.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "assets" / "samples"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset paths
CROHME_DATA_PATH = DATA_DIR / "crohme"
MATHWRITING_DATA_PATH = DATA_DIR / "mathwriting"

# Tokenizer settings
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "assets" / "vocab.txt"

# SymPy solver settings
MAX_SOLVING_TIME = 5.0  # Maximum seconds to spend on solving an equation


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Input/Output parameters
    input_size: Tuple[int, int] = (512, 2048)  # Height, Width - wide format for equations
    max_sequence_length: int = 150  # Maximum LaTeX sequence length
    
    # CNN Encoder Parameters
    cnn_filters: List[int] = (32, 64, 128, 256)
    cnn_kernel_size: Tuple[int, int] = (3, 3)
    cnn_pool_size: Tuple[int, int] = (2, 2)
    
    # RNN Model Parameters
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.1
    rnn_bidirectional: bool = True
    embedding_dim: int = 128
    
    # Transformer Model Parameters
    transformer_d_model: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    transformer_dim_feedforward: int = 512
    
    # Attention Parameters
    attention_dim: int = 256
    
    # TFLite Optimization Parameters
    quantization_aware_training: bool = True
    pruning_threshold: float = 0.01


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    data_dir: Path
    batch_size: int = 32
    input_size: Tuple[int, int] = (512, 2048)  # Wide format for equations
    model_type: str = "rnn"  # 'rnn' or 'transformer'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    rotation_range: float = 5.0
    scale_range: float = 0.1
    translation_range: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    checkpoint_dir: Path = Path("./checkpoints")
    vocab_file: Optional[Path] = None
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class DeviceConfig:
    """Configuration for target Supernote device."""
    # Rockchip RK3566 Specifications
    cpu: str = "Quad-core ARM Cortex-A55 @ 1.8 GHz"
    gpu: str = "Mali-G52"
    npu: str = "1 TOPS Neural Processing Unit"
    ram: str = "4 GB"
    screen_resolution: Tuple[int, int] = (1404, 1872)  # pixels
    screen_dpi: int = 226  # dots per inch
    screen_type: str = "E-ink"
    
    # TFLite Optimization Parameters
    quantization_level: str = "INT8"  # Options: "FLOAT32", "FLOAT16", "INT8"
    operator_support: List[str] = ("ADD", "CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX")
    max_model_size_mb: int = 25  # Target maximum model size in MB
    target_latency_ms: int = 500  # Target inference latency in milliseconds
    
    # Display Parameters
    max_rendering_time_ms: int = 200  # Maximum time for rendering in milliseconds
    
    # Battery Considerations
    max_power_consumption_w: float = 2.0  # Maximum continuous power in watts
```

## Docker Integration

The Dockerfile is used to create a Python 3.7 environment with the necessary dependencies for PosFormer.

```dockerfile
FROM python:3.7-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch and related packages with specific versions
RUN pip install --no-cache-dir torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir pytorch-lightning==1.4.9 torchmetrics==0.6.0
RUN pip install --no-cache-dir einops pillow==8.4.0 matplotlib sympy numpy opencv-python

# Set PYTHONPATH to include PosFormer
ENV PYTHONPATH="/app:/posformer:${PYTHONPATH}"

# Set a volume to mount the PosFormer directory
VOLUME ["/posformer", "/app"]

# Command to run when container starts
CMD ["/bin/bash"]
```

## Running the Project

To test the PosFormer model with example handwritten math expressions:

1. First, build the Docker container:
   ```bash
   docker build -t posformer-env .
   ```

2. Run the tests on a specific image:
   ```bash
   docker run -v "/path/to/PosFormer-main:/posformer" -v "/path/to/supernote-math:/app" posformer-env python /app/scripts/test_with_examples.py --image /app/assets/20250422_073317_Page_1.png --model /posformer/lightning_logs/version_0/checkpoints/best.ckpt
   ```

This will process the handwritten math expression in the image, recognize the LaTeX representation, and attempt to solve it.

## Technical Challenges and Solutions

### 1. NumPy API Version Mismatch
When setting up the Docker environment, we encountered an API version mismatch between PyTorch and NumPy:

```
RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd
```

Solution: We need to install a compatible version of NumPy (version 1.17.3) before installing PyTorch, as the PyTorch binary was compiled against NumPy API version 0xe (14).

### 2. Python 3.7 on arm64
PosFormer requires Python 3.7, but the Apple arm64 architecture doesn't have official Python 3.7 packages available through normal channels.

Solution: We used Docker to create an isolated environment with the correct Python version and package versions, allowing us to run the PosFormer model regardless of the host system's Python version.

## Next Steps

1. **Fine-tuning**: Fine-tune the PosFormer model on more Supernote-specific handwriting samples for improved accuracy.
2. **API Integration**: Create a RESTful API to integrate with Supernote's software directly.
3. **Performance Optimization**: Profile and optimize the model for better performance on the Supernote's hardware.
4. **Expanded Math Support**: Add support for more complex mathematical expressions and symbolic manipulation.
```