#!/bin/bash
# Setup script to create the PosFormer environment with the correct dependencies

# Script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Creating PosFormer environment with Python 3.7..."

# Create a virtual environment
python3.7 -m venv PosFormerEnv
source PosFormerEnv/bin/activate

# Install PyTorch and dependencies as specified in PosFormer's README
echo "Installing PyTorch and dependencies..."
pip install --upgrade pip
pip install torch==1.8.1 torchvision==0.9.1
pip install pytorch-lightning==1.4.9 torchmetrics==0.6.0
pip install einops
pip install pillow==8.4.0
pip install matplotlib
pip install sympy
pip install numpy

# Add the PosFormer repository to PYTHONPATH
POSFORMER_PATH="$DIR/../PosFormer-main"
echo "Adding PosFormer to PYTHONPATH..."
echo "export PYTHONPATH=\"$POSFORMER_PATH:\$PYTHONPATH\"" >> PosFormerEnv/bin/activate

echo "Environment setup complete!"
echo "To activate, run: source PosFormerEnv/bin/activate"