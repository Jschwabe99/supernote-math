# Supernote Math Recognition System

An offline math recognition, solving, and rendering system for the Supernote Nomad e-ink tablet.

## Project Overview

This system enables the Supernote Nomad to detect, recognize, solve, and render handwritten mathematical expressions using a CNN+RNN/Transformer architecture with Metal acceleration on Apple Silicon.

### Current Development Focus

The current development focus is on optimizing the image preprocessing pipeline to standardize handwritten math expressions from different datasets (CROHME and MathWriting) into a consistent binary format suitable for efficient neural network training.

Key achievements:
- Implemented optimized preprocessing function with dataset-specific handling
- Created disk-based caching system using NumPy arrays for efficient storage
- Built extensive testing framework with visual comparison capabilities
- Added parallel processing support for handling large datasets
- Successfully verified preprocessing works correctly with both datasets
- Prepared infrastructure for processing the full ~500K image dataset

### Key Components

1. **Math Detection**: Identifies areas containing math expressions
2. **Recognition**: Converts handwritten math to LaTeX using CNN+RNN or CNN+Transformer models
3. **Symbolic Solver**: Processes LaTeX, solves equations/expressions using SymPy
4. **Handwriting Renderer**: Renders solutions in natural handwriting style

## Directory Structure

```
supernote-math/
├── app/                                # Application integration
│   ├── __init__.py                     
│   ├── detection.py                    # Math region detection module
│   └── pipeline.py                     # End-to-end recognition pipeline
│
├── cli/                                # Command-line interfaces
│   ├── __init__.py                     
│   ├── benchmark.py                    # Performance benchmarking tool
│   ├── convert.py                      # Model conversion for TFLite
│   ├── run.py                          # Main entry point for execution
│   └── train.py                        # Training script for models
│
├── core/                               # Core system functionality
│   ├── __init__.py                     
│   ├── data/                           # Data handling
│   │   ├── __init__.py                 
│   │   ├── augmentations.py            # Data augmentation pipeline
│   │   └── loaders.py                  # Dataset loaders (CROHME, MathWriting)
│   │
│   ├── model/                          # ML Models
│   │   ├── __init__.py                 
│   │   ├── recognition.py              # CNN+RNN/Transformer model
│   │   └── tokenizer.py                # LaTeX tokenization
│   │
│   ├── render/                         # Output rendering
│   │   ├── __init__.py                 
│   │   └── handwriting.py              # Handwriting-style renderer
│   │
│   └── solver/                         # Math solving
│       ├── __init__.py                 
│       └── sympy_solver.py             # SymPy-based expression solver
│
├── utils/                              # Shared utilities
│   ├── __init__.py                     
│   ├── cli_utils.py                    # CLI helpers
│   ├── device_utils.py                 # Device-specific optimizations
│   └── logging_utils.py                # Logging setup
│
├── tests/                              # Test suite
│   ├── __init__.py                     
│   ├── app/                            # App tests
│   │   └── __init__.py                 
│   └── core/                           # Core module tests
│       ├── __init__.py                 
│       ├── data/                       # Data module tests
│       │   ├── __init__.py             
│       │   └── test_loaders.py         # Test dataset loaders
│       ├── model/                      # Model tests
│       │   └── __init__.py             
│       ├── render/                     # Renderer tests
│       │   └── __init__.py             
│       └── solver/                     # Solver tests
│           ├── __init__.py             
│           └── test_sympy_solver.py    # Test symbolic solver
│
├── tools/                              # External tools
│   └── inkml2img/                      # Tool to convert InkML to images
│
├── assets/                             # Project assets
│   ├── fonts/                          # Handwriting fonts
│   │   └── Kalam/                      # Kalam handwriting font
│   └── samples/                        # Sample datasets
│       ├── crohme/                     # CROHME dataset
│       └── mathwriting/                # MathWriting dataset
│
├── scripts/                            # Utility scripts
│   ├── metal_check.py                  # Check Metal GPU availability
│   └── metal_test.py                   # Test Metal performance
│
├── config.py                           # Centralized configuration
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installation
├── .flake8                             # Flake8 configuration
└── pytest.ini                          # Pytest configuration
```

## Architecture

The system uses a CNN + RNN/Transformer model for handwritten math recognition:

- **CNN Backbone**: Custom CNN architecture optimized for Metal GPU on Apple Silicon
- **Sequence Modeling**: Choice of bidirectional LSTM with attention or Transformer
- **Model Optimization**: TFLite quantization and optimization for the RK3566 NPU
- **Symbolic Engine**: SymPy for parsing, solving, and evaluating expressions
- **Rendering**: Vector-based handwriting renderer using the Kalam font library

## Datasets

- **MathWriting 2024**: 230,000+ human-written formulas plus isolated symbols
- **CROHME 2019**: Competition on Recognition of Handwritten Mathematical Expressions

## Setup and Installation

1. Clone this repository and navigate to the project directory

2. Create and activate a Python 3.9 virtual environment:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download datasets:
   - MathWriting dataset: https://huggingface.co/datasets/MathWriting/MathWriting
   - CROHME dataset: https://www.isical.ac.in/~crohme/CROHME_latextools.htm

## Usage

### Preprocessing Dataset Images

The first step is preprocessing the handwritten math images:

```bash
# Test preprocessing with visualization on a small sample
python scripts/test_preprocessing.py --height 512 --width 1024 --output-dir ./assets/test_output

# Process and save a batch of images from HuggingFace datasets
python scripts/test_preprocessing.py --use-huggingface --test-bulk --bulk-samples 100 \
  --height 512 --width 1024 --num-workers 8 --save-processed

# Look for wide formulas to test aspect ratio handling
python scripts/test_preprocessing.py --find-wide-formulas --num-samples 20
```

### Visualizing Preprocessed Images

Once preprocessing is complete, you can visualize and analyze the results:

```bash
# View sample preprocessed images with statistics
python scripts/visualize_preprocessed.py --data-dir ./preprocessed_data --num-samples 5

# Show only statistics without visualization 
python scripts/visualize_preprocessed.py --data-dir ./preprocessed_data --stats-only
```

### Training a model

```bash
python -m cli.train --data_dir preprocessed_data --epochs 100 --batch_size 32 --model_type rnn
```

### Converting to TFLite

```bash
python -m cli.convert --model_path checkpoints/model.h5 --output_path models/model.tflite --quantize
```

### Running inference

```bash
python -m cli.run --image_path input.png --model_path models/model.tflite
```

### Benchmarking

```bash
python -m cli.benchmark --model_path models/model.tflite --num_runs 100 --quantized
```

## Preprocessing Pipeline

The preprocessing pipeline transforms handwritten math images into a standardized format for model training:

### Process Flow

1. **Input**: Images from CROHME (white strokes on black) and MathWriting (black strokes on white)
2. **Grayscale Conversion**: Convert all images to grayscale for consistent processing
3. **Dataset-Specific Processing**:
   - CROHME: Invert white-on-black to black-on-white format
   - MathWriting: Enhance faint strokes using CLAHE and aggressive thresholding
4. **Binarization**: Convert to pure black (0) and white (255) using optimized thresholding
5. **Content Detection**: Find ink strokes and calculate bounding box with configurable margin
6. **Aspect Ratio Preservation**: Scale content while maintaining proper proportions
   - Special handling for extremely wide formulas
   - Intelligent scaling based on content aspect ratio
7. **Padding**: Center content in standard canvas size (512x1024)
8. **Output**: Binary image where ink=0 (black) and paper=255 (white)

### Performance Metrics

- **Processing Speed**: ~0.0027s per image with 8 workers
- **Memory Usage**: ~1GB for processing 20 images with 8 workers
- **Storage**: ~512KB per preprocessed image
- **Full Dataset**: ~250GB for ~500K images
- **Format**: NumPy arrays (.npy) with shape (512, 1024, 1) and dtype uint8

### Optimizations

- **LRU Caching**: Frequently used objects (CLAHE, morphological kernels)
- **Parallel Processing**: ThreadPoolExecutor for multi-core utilization
- **Memory Efficiency**: Chunked processing to limit RAM usage
- **Early Exit Paths**: Skip processing for invalid inputs
- **Disk-Based Caching**: Avoid reprocessing the same images
- **Custom Format Handling**: Optimized paths for different image formats

## Target Device

Supernote Nomad with Rockchip RK3566 SoC (1.8 GHz Quad-core ARM Cortex-A55, Mali-G52 GPU, 1 TOPS NPU).