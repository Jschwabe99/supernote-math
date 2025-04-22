# PosFormer Integration for Supernote Math Recognition

This document explains how the PosFormer model has been integrated into the Supernote Math project to enable handwritten mathematical expression recognition (HMER).

## Overview

PosFormer is a state-of-the-art model for handwritten math recognition, which introduces a dual-task approach that optimizes both expression and position recognition. It uses a novel "position forest" structure to parse and model the hierarchical relationships of symbols without needing extra annotations.

We've created a pipeline that:
1. Takes stroke data from the Supernote device
2. Converts strokes to images
3. Preprocesses the images to the format expected by PosFormer
4. Runs the PosFormer model to recognize the mathematical expressions
5. Outputs LaTeX that can be further processed or rendered

## Directory Structure

```
supernote-math/
├── app/
│   ├── __init__.py
│   ├── detection.py      # Math region detection
│   └── pipeline.py       # PosFormer integration pipeline
├── core/
│   ├── data/
│   │   └── loaders.py    # Image preprocessing for the model
│   └── solver/
│       └── sympy_solver.py  # Solving recognized expressions
├── scripts/
│   └── test_posformer_pipeline.py  # Test script for the pipeline
└── PosFormer-main/       # The PosFormer model (in parent directory)
    └── ...
```

## How to Use

### Quick Start: Running All Tests

The easiest way to test the entire pipeline is to use the provided run script:

```bash
cd /Users/julian/Coding/Supernote Project/supernote-math
./run_tests.sh
```

This will:
1. Test with simulated strokes (a generated quadratic equation)
2. Test with a full page of handwritten math expressions
3. Display the recognized LaTeX and solutions for each test

### Testing with Simulated Strokes

To test specifically with simulated stroke data:

```bash
python scripts/test_posformer_pipeline.py
```

This will:
1. Generate simulated strokes for a quadratic equation
2. Convert the strokes to an image
3. Run the PosFormer recognition model
4. Display the recognized LaTeX

### Testing with a Page of Math Expressions

To test with the included test page containing multiple expressions:

```bash
python scripts/test_with_examples.py --test-page
```

This will:
1. Load the test page image
2. Segment it into individual expressions
3. Recognize each expression
4. Display the results with bounding boxes

### Testing with Your Own Images

You can also test with your own images:

```bash
# For a single expression image
python scripts/test_with_examples.py --image /path/to/math/image.png

# For a page with multiple expressions
python scripts/test_with_examples.py --image /path/to/math/page.png --page
```

### Using a Custom Model

If you have a different checkpoint for the PosFormer model:

```bash
# With the run script
./run_tests.sh --model /path/to/model/checkpoint.ckpt

# Or with individual scripts
python scripts/test_with_examples.py --test-page --model /path/to/model/checkpoint.ckpt
```

## Integration into Your Application

To integrate the math recognition pipeline into your application:

```python
from app.pipeline import MathRecognitionPipeline, Stroke

# Initialize the pipeline
pipeline = MathRecognitionPipeline(
    posformer_model_path="/path/to/checkpoint.ckpt"  # Optional
)

# Create strokes from user input
strokes = []
new_stroke = Stroke()
new_stroke.add_point(x=100, y=100, pressure=1.0)
new_stroke.add_point(x=200, y=200, pressure=1.0)
strokes.append(new_stroke)

# Recognize and solve
result = pipeline.process_strokes(strokes)

# Get the results
latex = result['latex']
print(f"Recognized: {latex}")

if result['solution']['solved']:
    print(f"Solution: {result['solution']['result_latex']}")
```

## Working with JSON Data from Supernote

If you have JSON data from the Supernote device:

```python
from app.pipeline import MathRecognitionPipeline

# Initialize the pipeline
pipeline = MathRecognitionPipeline()

# Parse JSON data from Supernote
with open('supernote_strokes.json', 'r') as f:
    json_data = json.load(f)

# Extract strokes
strokes = pipeline.extract_strokes_from_json(json_data)

# Process strokes
result = pipeline.process_strokes(strokes)
print(f"Recognized: {result['latex']}")
```

## Customizing the Pipeline

### Adjusting Preprocessing

The preprocessing parameters can be customized in the `recognize` method of `PosFormerRecognizer`:

```python
processed = preprocess_image(
    image=image,
    target_resolution=(256, 1024),  # Change resolution
    invert=False,                   # Invert colors if needed
    normalize=False,                # Normalization option
    crop_margin=0.05,               # Adjust margin around content
    min_margin_pixels=20            # Minimum margin
)
```

### Beam Search Parameters

The beam search parameters can be adjusted in the `config` dictionary in the `PosFormerRecognizer` class:

```python
self.config = {
    # ... other parameters ...
    'beam_size': 10,             # Increase for better results, slower inference
    'max_len': 200,              # Maximum sequence length
    'alpha': 1.0,                # Length penalty
    'early_stopping': True,      # Stop when best sequence is found
    'temperature': 1.0,          # Sampling temperature
}
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- PIL (Pillow)
- SymPy (for equation solving)
- Matplotlib (for visualization)

## Future Improvements

1. Optimize performance for real-time recognition
2. Improve handling of different stroke styles and pressures
3. Add support for more complex mathematical notation
4. Implement real-time feedback during writing
5. Create a stroke library for handwriting-style rendering of solutions