#!/bin/bash
# Script to build and run the PosFormer Docker container for testing

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

# Run a test case with one of the PNG files
echo "Running container with test image..."
docker run --rm -it \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer python /app/scripts/test_posformer_pipeline.py \
  --image /app/assets/20250422_073317_Page_1.png \
  --model /posformer/lightning_logs/version_0/checkpoints/best.ckpt \
  --no-show

# Final message
echo "Test completed. Check the output above for any errors."