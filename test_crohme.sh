#!/bin/bash
# Script to run CROHME testing with Python virtual environment

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Determine optimal CPU core count
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
WORKER_CORES=$((CPU_CORES - 1))  # Leave one core for system
if [ "$WORKER_CORES" -lt 1 ]; then
    WORKER_CORES=1
fi

echo "Running with $WORKER_CORES CPU cores using Python virtual environment..."

# Activate the virtual environment
source "$DIR/activate_env.sh"

# Run the optimized script
python -u "$DIR/scripts/crohme/optimized_process_crohme.py" \
  --posformer_dir "$POSFORMER_DIR" \
  --crohme_dir "$DIR/CROHME-full" \
  --output_dir "$DIR/assets/crohme_results" \
  --sample_size 20 \
  --no_invert \
  --num_workers "$WORKER_CORES"

echo "Testing completed! Check assets/crohme_results for results."