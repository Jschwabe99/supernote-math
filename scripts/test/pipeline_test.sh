#!/bin/bash
# Test script for the full PosFormer pipeline

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Run the full pipeline test
echo "Testing full pipeline with image..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer \
  python "/app/scripts/test_posformer_pipeline.py" \
  --image "/app/assets/20250422_073317_Page_1.png" \
  --model "/posformer/lightning_logs/version_0/checkpoints/best.ckpt" \
  --no-show