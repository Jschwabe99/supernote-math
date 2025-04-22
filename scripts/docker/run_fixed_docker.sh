#!/bin/bash
# Comprehensive fix for PosFormer Docker integration
# This script builds the Docker image, applies model architecture fixes,
# and runs inference on all test images

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"
CHECKPOINT_PATH="$POSFORMER_DIR/lightning_logs/version_0/checkpoints/best.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
  exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t supernote-posformer -f "$DIR/docker-scripts/Dockerfile" "$DIR"

# Make sure the build succeeded
if [ $? -ne 0 ]; then
  echo "Error: Docker build failed"
  exit 1
fi

# First run the diagnostic tool to get detailed information
echo "Running diagnostics..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer python /app/scripts/debug_posformer.py \
  --model /posformer/lightning_logs/version_0/checkpoints/best.ckpt \
  --image /app/assets/20250422_073317_Page_1.png

# Apply model architecture fixes
echo "Applying model architecture fixes..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer python /app/scripts/model_fix.py \
  --checkpoint /posformer/lightning_logs/version_0/checkpoints/best.ckpt

# Process all test images
echo "Processing all test images..."
for i in {1..9}; do
  echo "Processing image $i..."
  docker run --rm \
    -v "$POSFORMER_DIR:/posformer" \
    -v "$DIR:/app" \
    supernote-posformer python /app/scripts/test_posformer_pipeline.py \
    --image "/app/assets/20250422_073317_Page_$i.png" \
    --model /posformer/lightning_logs/version_0/checkpoints/best.ckpt \
    --no-show
done

echo "All images processed. Check the output above for results."