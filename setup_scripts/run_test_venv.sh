#!/bin/bash
# Script to run CROHME testing using the virtual environment

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Copy the optimized script to the main directory
cp "$DIR/scripts/crohme/optimized_process_crohme.py" "$DIR/optimized_process_crohme.py"

# Determine optimal CPU core count
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
WORKER_CORES=$((CPU_CORES / 2))  # Use half cores for stability
if [ "$WORKER_CORES" -lt 1 ]; then
    WORKER_CORES=1
fi

echo "Running with $WORKER_CORES CPU cores using virtual environment..."

# Activate the virtual environment
source "$DIR/posformer_venv/bin/activate"

# Set environment variables
export OMP_NUM_THREADS=$WORKER_CORES
export OPENBLAS_NUM_THREADS=$WORKER_CORES
export MKL_NUM_THREADS=$WORKER_CORES
export VECLIB_MAXIMUM_THREADS=$WORKER_CORES
export NUMEXPR_NUM_THREADS=$WORKER_CORES
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$DIR:$POSFORMER_DIR:$PYTHONPATH"

# Verify Python and package versions
echo "Running with Python $(python --version)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import PIL; print('Pillow version:', PIL.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'MPS not supported')"

# Run the optimized Python script
cd "$DIR"
python -u optimized_process_crohme.py \
  --posformer_dir "$POSFORMER_DIR" \
  --crohme_dir "$DIR/CROHME-full" \
  --output_dir "$DIR/assets/crohme_results" \
  --sample_size 10 \
  --no_invert \
  --num_workers $WORKER_CORES

# Clean up the copied script
rm "$DIR/optimized_process_crohme.py"

# Deactivate virtual environment
deactivate

echo "Testing completed! Check assets/crohme_results for results."