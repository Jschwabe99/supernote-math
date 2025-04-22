#!/bin/bash
# Run dataset tools with no timeout
# Usage: ./run_dataset_tools.sh [explore|download] [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Parse the command
COMMAND=$1
shift

case "$COMMAND" in
  explore)
    echo "Running explore_dataset.py with arguments: $@"
    python "$SCRIPT_DIR/explore_dataset.py" "$@"
    ;;
  download)
    echo "Running download_datasets.py with arguments: $@"
    python "$SCRIPT_DIR/download_datasets.py" "$@"
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Usage: $0 [explore|download] [options]"
    echo ""
    echo "Examples:"
    echo "  $0 explore --dataset \"andito/mathwriting-google\" --num_examples 5 --output_dir \"$PROJECT_DIR/assets/samples/mathwriting\""
    echo "  $0 explore --dataset \"Neeze/CROHME-full\" --num_examples 5 --output_dir \"$PROJECT_DIR/assets/samples/crohme\""
    echo "  $0 download --dataset mathwriting --sample_size 10 --output_dir \"$PROJECT_DIR/assets/samples/mathwriting\" --save_examples"
    exit 1
    ;;
esac