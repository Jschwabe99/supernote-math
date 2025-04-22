#!/bin/bash
# Test script for the full PosFormer pipeline with OpenBLAS fix

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run the full pipeline test with OpenBLAS thread settings
echo "Testing full pipeline with image..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  -e "VECLIB_MAXIMUM_THREADS=1" \
  -e "NUMEXPR_NUM_THREADS=1" \
  supernote-posformer \
  python "/app/scripts/test_posformer_pipeline.py" \
  --image "/app/assets/20250422_073317_Page_1.png" \
  --model "/posformer/lightning_logs/version_0/checkpoints/best.ckpt" \
  --no-show