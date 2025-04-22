"""
Centralized configuration for the Supernote Math Recognition project.
This module contains configuration classes for all components of the system.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "assets" / "samples"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset paths
CROHME_DATA_PATH = DATA_DIR / "crohme"
MATHWRITING_DATA_PATH = DATA_DIR / "mathwriting"

# Tokenizer settings
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "assets" / "vocab.txt"

# SymPy solver settings
MAX_SOLVING_TIME = 5.0  # Maximum seconds to spend on solving an equation


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Input/Output parameters
    input_size: Tuple[int, int] = (512, 2048)  # Height, Width - wide format for equations
    max_sequence_length: int = 150  # Maximum LaTeX sequence length
    
    # CNN Encoder Parameters
    cnn_filters: List[int] = (32, 64, 128, 256)
    cnn_kernel_size: Tuple[int, int] = (3, 3)
    cnn_pool_size: Tuple[int, int] = (2, 2)
    
    # RNN Model Parameters
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.1
    rnn_bidirectional: bool = True
    embedding_dim: int = 128
    
    # Transformer Model Parameters
    transformer_d_model: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    transformer_dim_feedforward: int = 512
    
    # Attention Parameters
    attention_dim: int = 256
    
    # TFLite Optimization Parameters
    quantization_aware_training: bool = True
    pruning_threshold: float = 0.01


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    data_dir: Path
    batch_size: int = 32
    input_size: Tuple[int, int] = (512, 2048)  # Wide format for equations
    model_type: str = "rnn"  # 'rnn' or 'transformer'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    rotation_range: float = 5.0
    scale_range: float = 0.1
    translation_range: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    checkpoint_dir: Path = Path("./checkpoints")
    vocab_file: Optional[Path] = None
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class DeviceConfig:
    """Configuration for target Supernote device."""
    # Rockchip RK3566 Specifications
    cpu: str = "Quad-core ARM Cortex-A55 @ 1.8 GHz"
    gpu: str = "Mali-G52"
    npu: str = "1 TOPS Neural Processing Unit"
    ram: str = "4 GB"
    screen_resolution: Tuple[int, int] = (1404, 1872)  # pixels
    screen_dpi: int = 226  # dots per inch
    screen_type: str = "E-ink"
    
    # TFLite Optimization Parameters
    quantization_level: str = "INT8"  # Options: "FLOAT32", "FLOAT16", "INT8"
    operator_support: List[str] = ("ADD", "CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX")
    max_model_size_mb: int = 25  # Target maximum model size in MB
    target_latency_ms: int = 500  # Target inference latency in milliseconds
    
    # Display Parameters
    max_rendering_time_ms: int = 200  # Maximum time for rendering in milliseconds
    
    # Battery Considerations
    max_power_consumption_w: float = 2.0  # Maximum continuous power in watts