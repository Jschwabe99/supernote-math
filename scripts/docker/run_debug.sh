#!/bin/bash
# Script to run diagnostic script inside Docker container

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"
CHECKPOINT_PATH="$POSFORMER_DIR/lightning_logs/version_0/checkpoints/best.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
  exit 1
fi

# Build the Docker image if needed
if [[ "$(docker images -q supernote-posformer 2> /dev/null)" == "" ]]; then
  echo "Building Docker image..."
  docker build -t supernote-posformer -f "$DIR/docker-scripts/Dockerfile" "$DIR"

  # Make sure the build succeeded
  if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
  fi
fi

# Run the diagnostic script in the container
echo "Running diagnostic script inside container..."
docker run --rm -it \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  supernote-posformer python /app/scripts/debug_posformer.py \
  --model /posformer/lightning_logs/version_0/checkpoints/best.ckpt \
  --image /app/assets/20250422_073317_Page_1.png

# Final message
echo "Diagnostics completed. Check the output above for errors."