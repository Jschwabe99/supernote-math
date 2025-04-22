#!/bin/bash
# Script to install dependencies in the Docker container

set -e  # Exit on error

echo "=== Step 1: Installing system dependencies ==="
apt-get update
apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 git

echo "=== Step 2: Installing specific NumPy version ==="
# We need NumPy 1.20.0 which is compatible with PyTorch 1.8.1's API version 0xe
pip install --no-cache-dir numpy==1.20.0

echo "=== Step 3: Installing PyTorch ==="
pip install --no-cache-dir torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Verify PyTorch works with NumPy
python -c 'import torch; print(f"PyTorch version: {torch.__version__}")'

echo "=== Step 4: Installing PosFormer dependencies ==="
pip install --no-cache-dir pytorch-lightning==1.4.9 torchmetrics==0.6.0 

echo "=== Step 5: Installing other dependencies ==="
pip install --no-cache-dir einops pillow==8.4.0 matplotlib sympy opencv-python

echo "=== All dependencies installed successfully ==="
python --version
python -c 'import torch; print(f"PyTorch version: {torch.__version__}")'
python -c 'import numpy; print(f"NumPy version: {numpy.__version__}")'

echo "=== Environment setup complete ==="