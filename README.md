# Supernote Math Recognition Project

A mathematical expression recognition system for the Supernote Nomad e-ink tablet, using the PosFormer model.

## Project Overview

This project implements an efficient pipeline for recognizing handwritten mathematical expressions from Supernote devices. It handles the preprocessing of images to standardize the format required by the PosFormer model and provides a Docker-based solution for consistent execution across platforms.

## Features

- Enhanced preprocessing pipeline for handwritten math images
- Support for both Supernote (black-on-white) and CROHME (white-on-black) formats
- Docker container for isolated execution
- Integration with the PosFormer model for mathematical expression recognition
- Comprehensive testing and benchmarking tools
- CROHME dataset evaluation capabilities

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

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/supernote-math.git
cd supernote-math
```

2. Build the Docker container:
```bash
./scripts/docker/get-docker.sh
```

### Usage

#### Process Supernote Images
```bash
./scripts/solution/final_working_solution.sh
```

#### Test with CROHME Dataset
```bash
./scripts/crohme/run_crohme_docker.sh
```

#### Run Enhanced Preprocessing
```bash
./scripts/preprocessing/fixed_enhanced_preprocessing.sh
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