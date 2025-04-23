#!/bin/bash
# Source this file to activate the environment
cd "$(dirname "$0")"
source posformer_venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/../PosFormer-main:$PYTHONPATH"
echo "Supernote Math environment activated! PYTHONPATH includes project directory and PosFormer-main"
