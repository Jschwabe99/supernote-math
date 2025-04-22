"""Command-line interface utilities."""

import typer
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum
from .logging_utils import get_logger

logger = get_logger(__name__)

class ModelType(str, Enum):
    """Supported model architectures."""
    RNN = "rnn"
    TRANSFORMER = "transformer"

def load_config_file(config_file: Optional[Path]) -> Dict[str, Any]:
    """Load a configuration file.
    
    Args:
        config_file: Path to a JSON or YAML configuration file
        
    Returns:
        Dictionary of configuration values
    """
    if not config_file:
        return {}
    
    config_path = Path(config_file)
    if not config_path.exists():
        logger.warning(f"Config file {config_file} does not exist. Using defaults.")
        return {}
    
    try:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            import json
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Unsupported config file format: {config_path.suffix}")
            return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}