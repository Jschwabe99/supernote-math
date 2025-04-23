# Supernote Math Recognition & Solver

A desktop application that recognizes and solves handwritten mathematical expressions from the Supernote Nomad e-ink tablet, using the PosFormer model and SymPy.

## Project Overview

This project provides a macOS desktop application that recognizes handwritten mathematical expressions from Supernote PNG exports and solves them. It leverages the PosFormer model for mathematical expression recognition and SymPy for symbolic computation. The core functionality includes:

1. Image preprocessing to standardize exported Supernote math images
2. Mathematical expression recognition using PosFormer AI
3. Conversion of recognized expressions to SymPy format
4. Solving equations or evaluating arithmetic expressions
5. Presenting the solution to the user

## Features

- Simple Mac desktop interface for importing Supernote equation images
- Enhanced preprocessing pipeline for handwritten math images
- Support for both Supernote (black-on-white) and CROHME (white-on-black) formats
- Equation solving capabilities using SymPy
- Arithmetic evaluation for expressions without variables
- Docker-based development environment

## Project Structure

```
supernote-math/
├── app/                # Application code for deployment
├── assets/             # Test outputs and sample images
├── core/               # Core functionality modules
│   └── data/           # Data processing modules
├── docs/               # Project documentation
├── scripts/            # Organized scripts directory
│   ├── crohme/         # CROHME dataset processing
│   ├── docker/         # Docker-related scripts
│   ├── preprocessing/  # Image preprocessing scripts
│   ├── solution/       # Final working solution scripts
│   └── test/           # Testing scripts
└── ...
```

## Quick Start

### Development Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/supernote-math.git
cd supernote-math
```

2. Build the Docker container (for development and testing):
```bash
./build_container.sh
```

### Running the Application

#### Mac Desktop Application
```bash
# Run the desktop app
python app/main.py
```

#### Development and Testing

##### Test with Sample Supernote Images
```bash
./scripts/solution/final_working_solution.sh
```

##### Evaluate with CROHME Dataset
```bash
./run_long_crohme_test.sh
```

##### Test Equation Solving Module
```bash
python -m tests.core.solver.test_sympy_solver
```

## Performance

- Processing time: ~0.05s per image with Docker container
- Recognition accuracy on Supernote samples: Good for clear handwriting
- CROHME dataset baseline accuracy: 10% without fine-tuning
- Memory usage: Optimized for Docker container execution
- Preprocessing time: ~0.003s per image

## Documentation

For more detailed information, please see the following documentation files:

- [POSFORMER_INTEGRATION.md](docs/POSFORMER_INTEGRATION.md) - Details on PosFormer model integration
- [POSFORMER_FINAL_SOLUTION.md](docs/POSFORMER_FINAL_SOLUTION.md) - Final working solution documentation
- [DOCKER_SOLUTION.md](docs/DOCKER_SOLUTION.md) - Docker implementation details
- [PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) - Comprehensive project documentation

## License

[MIT License](LICENSE)

## Acknowledgments

- PosFormer model for mathematical expression recognition
- CROHME dataset for testing and evaluation
- MathWriting dataset for additional training data