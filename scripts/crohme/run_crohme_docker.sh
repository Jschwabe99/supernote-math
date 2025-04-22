#!/bin/bash
# Script to run CROHME testing in Docker

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Create output directory
mkdir -p "$DIR/assets/crohme_results"

# Create a copy of the Python script in the main directory
# (This helps with Docker volume mounting)
cp "$DIR/scripts/process_crohme.py" "$DIR/process_crohme.py"

# Run the Docker container
echo "Running CROHME dataset test in Docker..."
docker run --rm \
  -v "$POSFORMER_DIR:/posformer" \
  -v "$DIR:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer \
  bash -c "
  # Install required dependencies
  pip install pandas pyarrow matplotlib tqdm --quiet

  # Run the Python script
  cd /app
  python process_crohme.py --posformer_dir /posformer --crohme_dir /app/CROHME-full --output_dir /app/assets/crohme_results --sample_size 10 --no_invert
  "

# Clean up the copied script
rm "$DIR/process_crohme.py"

echo "Testing completed! Check the 'assets/crohme_results' directory for results."