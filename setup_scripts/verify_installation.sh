#!/bin/bash
# Script to verify the virtual environment installation is working correctly

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
POSFORMER_DIR="$DIR/../PosFormer-main"

# Activate the virtual environment
source "$DIR/posformer_venv/bin/activate"
export PYTHONPATH="$DIR:$POSFORMER_DIR:$PYTHONPATH"

echo "========== Python and Package Versions =========="
echo "Python: $(python --version)"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import einops; print(f'einops: {einops.__version__}')"
python -c "import sympy; print(f'SymPy: {sympy.__version__}')"

echo -e "\n========== PyTorch Hardware Acceleration =========="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
python -c "import torch; print(f'Using device: {\"mps\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\"}')"

echo -e "\n========== PosFormer Model Imports =========="
if [ -d "$POSFORMER_DIR" ]; then
    python -c "import sys; sys.path.append('$POSFORMER_DIR'); 
try:
    from Pos_Former.pos_former import PosFormer
    print('✅ PosFormer model imports successfully')
except Exception as e:
    print(f'❌ Error importing PosFormer model: {e}')"
else
    echo "❌ PosFormer directory not found at $POSFORMER_DIR"
    echo "Please make sure the PosFormer repository is cloned to the correct location"
fi

echo -e "\n========== Project Files Check =========="
[ -f "$DIR/core/solver/sympy_solver.py" ] && echo "✅ Equation solver module found" || echo "❌ Equation solver module not found"
[ -f "$DIR/app/pipeline.py" ] && echo "✅ Pipeline module found" || echo "❌ Pipeline module not found"
[ -f "$DIR/app/detection.py" ] && echo "✅ Detection module found" || echo "❌ Detection module not found"
[ -d "$DIR/assets" ] && echo "✅ Assets directory found" || echo "❌ Assets directory not found"

echo -e "\n========== Dataset Check =========="
[ -d "$DIR/CROHME-full" ] && echo "✅ CROHME dataset found" || echo "❓ CROHME dataset not found (optional for testing)"

# Deactivate virtual environment
deactivate

echo -e "\nVerification complete!"
echo "If all checks passed, your virtual environment is set up correctly."
echo "If there were any issues, please address them before proceeding."