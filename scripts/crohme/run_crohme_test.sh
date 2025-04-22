#!/bin/bash
# Script to set up a venv and run CROHME testing

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Set up virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv "$DIR/venv_crohme"
source "$DIR/venv_crohme/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install pandas pyarrow matplotlib tqdm pillow opencv-python torch pytorch-lightning

# Run the Python script
echo "Running CROHME dataset test..."
python "$DIR/scripts/process_crohme.py" \
  --posformer_dir "$POSFORMER_DIR" \
  --crohme_dir "$DIR/CROHME-full" \
  --output_dir "$DIR/assets/crohme_results" \
  --sample_size 10

# Deactivate virtual environment
deactivate

echo "Testing completed! Check the 'assets/crohme_results' directory for results."