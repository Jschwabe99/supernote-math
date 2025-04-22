#!/bin/bash
# Run PosFormer tests with the correct Python 3.7 environment
# This script activates the PosFormer environment and runs tests

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if the PosFormer environment exists
if [ ! -d "$DIR/PosFormerEnv" ]; then
  echo "Error: PosFormer environment not found at $DIR/PosFormerEnv"
  echo "Creating environment first..."
  "$DIR/setup_posformer_env.sh"
fi

echo "Activating PosFormer environment..."
source "$DIR/PosFormerEnv/bin/activate"

# Print debug info
echo "Using Python: $(which python)"
echo "PyTorch available:" 
python -c "try:
    import torch
    print(f'Yes - PyTorch {torch.__version__}')
except Exception as e:
    print(f'No - {e}')"

# Add script directory to PYTHONPATH
export PYTHONPATH="$DIR:$PYTHONPATH"

# Run specific test script if provided
if [ "$1" == "--test-mock" ]; then
  echo -e "\nRunning mock tests...\n"
  python "$DIR/scripts/test_mock.py"
elif [ "$1" == "--image" ]; then
  echo -e "\nTesting with specific image...\n"
  if [ -z "$2" ]; then
    echo "Error: Please provide an image path"
    exit 1
  fi
  python "$DIR/scripts/test_with_examples.py" --image "$2"
elif [ "$1" == "--test-page" ]; then
  echo -e "\nTesting with example page...\n"
  python "$DIR/scripts/test_with_examples.py" --test-page
else
  # Run the tests with the test page
  echo -e "\nRunning test with example page...\n"
  python "$DIR/scripts/test_with_examples.py" --test-page
fi

echo "Deactivating environment..."
deactivate