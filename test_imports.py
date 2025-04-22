#!/usr/bin/env python
"""Test importing dependencies."""

import sys
import os

# Print Python path
print("Python path:")
for p in sys.path:
    print(f"  - {p}")

print("\nTrying to import packages:")

# Try to import packages
try:
    import einops
    print(f"✅ einops {einops.__version__}")
except ImportError as e:
    print(f"❌ einops: {e}")

try:
    import torch
    print(f"✅ torch {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")

try:
    import pytorch_lightning
    print(f"✅ pytorch_lightning {pytorch_lightning.__version__}")
except ImportError as e:
    print(f"❌ pytorch_lightning: {e}")

try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

# Try to import from PosFormer
print("\nTrying to import from PosFormer:")

# Add PosFormer to path
posformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PosFormer-main')
sys.path.append(posformer_path)

try:
    from Pos_Former.utils import utils
    print("✅ Pos_Former.utils.utils")
except ImportError as e:
    print(f"❌ Pos_Former.utils.utils: {e}")

try:
    from Pos_Former.model import posformer
    print("✅ Pos_Former.model.posformer")
except ImportError as e:
    print(f"❌ Pos_Former.model.posformer: {e}")