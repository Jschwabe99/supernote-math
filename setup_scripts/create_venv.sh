#!/bin/bash
# Script to create a virtual environment with the same dependencies as the Docker container

echo "Creating virtual environment for Supernote Math project..."
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# Use Python 3.9 specifically
PYTHON_CMD="/opt/homebrew/bin/python3.9"
echo "Using Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv posformer_venv

# Activate virtual environment
echo "Activating virtual environment..."
source posformer_venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install NumPy first (critical)
echo "Installing NumPy 1.24.3 (compatible with Python 3.9)..."
pip install numpy==1.24.3

# Install PyTorch and torchvision
echo "Installing PyTorch 2.0.1 and torchvision 0.15.2 (compatible with Python 3.9)..."
pip install torch==2.0.1 torchvision==0.15.2

# Install the rest of the requirements
echo "Installing other dependencies..."
pip install pytorch-lightning==2.0.1 torchmetrics==0.11.4 Pillow==9.5.0
pip install opencv-python==4.7.0.72 einops==0.6.1 typer==0.9.0 matplotlib==3.7.2 sympy==1.12 tqdm

# Install additional dependencies for CROHME dataset processing
echo "Installing dataset processing dependencies..."
pip install pandas pyarrow

# Create activation script
cat > activate_env.sh << 'ACTIVATE'
#!/bin/bash
# Source this file to activate the environment
cd "$(dirname "$0")"
source posformer_venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/../PosFormer-main:$PYTHONPATH"
echo "Supernote Math environment activated! PYTHONPATH includes project directory and PosFormer-main"
ACTIVATE

chmod +x activate_env.sh

echo "Environment setup complete!"
echo "To activate the environment, run: source activate_env.sh"