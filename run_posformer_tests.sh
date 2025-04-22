#!/bin/bash
# Run all PosFormer pipeline tests
# This script activates the PosFormer virtual environment and runs the test suite

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the PosFormer environment
echo "Activating PosFormer environment..."
source "$DIR/PosFormerEnv/bin/activate"

# Run the tests
python "$DIR/scripts/run_posformer_tests.py" "$@"

# Deactivate virtual environment
echo "Deactivating environment..."
deactivate