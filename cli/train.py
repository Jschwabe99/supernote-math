"""Training script for math recognition model."""

import typer
import logging
import sys
import os
import tensorflow as tf
import numpy as np
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model.recognition import MathRecognitionModel
from core.model.tokenizer import MathTokenizer
from core.data.loaders import DataLoader
from core.data.augmentations import AugmentationPipeline
from utils.device_utils import configure_metal, get_compute_capability, set_training_memory_options
from config import TrainingConfig, ModelConfig, CROHME_DATA_PATH, MATHWRITING_DATA_PATH, PROJECT_ROOT

app = typer.Typer(help="Train the math recognition model")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_callbacks(checkpoint_dir: Path, patience: int = 10, reduce_lr_patience: int = 5):
    """Create callbacks for model training.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        patience: Number of epochs with no improvement to wait before early stopping
        reduce_lr_patience: Number of epochs with no improvement to wait before reducing LR
        
    Returns:
        List of callbacks for model training
    """
    callbacks = []
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Model checkpoint callback
    checkpoint_path = checkpoint_dir / "model_checkpoint_{epoch:02d}_{val_loss:.4f}.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard callback
    tensorboard_dir = checkpoint_dir / "logs" / f"run_{int(time.time())}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    return callbacks

@app.command()
def train(
    data_dir: Optional[Path] = typer.Option(None, help="Directory containing training data"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    model_type: str = typer.Option("rnn", help="Model architecture type ('rnn' or 'transformer')"),
    checkpoint_dir: Path = typer.Option(None, help="Directory to save model checkpoints"),
    dataset: str = typer.Option("mathwriting", help="Dataset to use ('crohme' or 'mathwriting')"),
    learning_rate: float = typer.Option(1e-3, help="Initial learning rate"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    output_dir: Optional[Path] = typer.Option(None, help="Directory for output files"),
    use_metal: bool = typer.Option(True, help="Use Metal GPU acceleration if available"),
    memory_limit_mb: Optional[int] = typer.Option(None, help="GPU memory limit in MB (None for no limit)")
):
    """Train the math recognition model."""
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level))
    
    # Configure Metal GPU if requested and available
    if use_metal:
        metal_configured = configure_metal()
        if metal_configured:
            # Set memory options
            set_training_memory_options(memory_limit=memory_limit_mb)
    
    # Get device information for logging
    device_info = get_compute_capability()
    logger.info(f"Compute capability: {device_info}")
    
    # Set default data directory if not provided
    if data_dir is None:
        if dataset.lower() == 'crohme':
            data_dir = CROHME_DATA_PATH
        else:
            data_dir = MATHWRITING_DATA_PATH
        logger.info(f"Using default data directory: {data_dir}")
    
    # Set default checkpoint directory if not provided
    if checkpoint_dir is None:
        checkpoint_dir = PROJECT_ROOT / "checkpoints" / f"{dataset}_{model_type}"
        logger.info(f"Using default checkpoint directory: {checkpoint_dir}")
    
    # Log command execution
    logger.info(f"Starting training with {model_type} architecture")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
    
    # Create configuration object
    training_config = TrainingConfig(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        learning_rate=learning_rate
    )
    
    model_config = ModelConfig()
    
    logger.info(f"Training configuration: {training_config}")
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = DataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        input_size=model_config.input_size,
        use_augmentation=True
    )
    
    # Get dataset splits
    logger.info(f"Loading {dataset} dataset splits...")
    if dataset.lower() == 'crohme':
        train_ds = data_loader.load_crohme_dataset('train')
        val_ds = data_loader.load_crohme_dataset('val')
        test_ds = data_loader.load_crohme_dataset('test')
    else:  # mathwriting
        train_ds = data_loader.load_mathwriting_dataset('train')
        val_ds = data_loader.load_mathwriting_dataset('val')
        test_ds = data_loader.load_mathwriting_dataset('test')
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = MathTokenizer()
    
    # Initialize model
    logger.info(f"Building {model_type} model...")
    model = MathRecognitionModel(
        model_config=model_config,
        tokenizer=tokenizer,
        model_type=model_type
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        patience=training_config.early_stopping_patience,
        reduce_lr_patience=training_config.reduce_lr_patience
    )
    
    # Train the model
    logger.info("Starting model training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training results
        logger.info(f"Training completed. Final validation loss: {history.history['val_loss'][-1]:.4f}")
        logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        logger.info(f"Model saved to {checkpoint_dir}")
        
        return history
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None

@app.command()
def evaluate(
    model_path: Path = typer.Option(..., help="Path to the trained model"),
    test_data: Optional[Path] = typer.Option(None, help="Path to test data"),
    dataset: str = typer.Option("mathwriting", help="Dataset to use ('crohme' or 'mathwriting')"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    output_dir: Optional[Path] = typer.Option(None, help="Directory for evaluation results"),
    use_metal: bool = typer.Option(True, help="Use Metal GPU acceleration if available")
):
    """Evaluate a trained model on test data."""
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level))
    
    # Configure Metal GPU if requested and available
    if use_metal:
        configure_metal()
    
    # Set default test data directory if not provided
    if test_data is None:
        if dataset.lower() == 'crohme':
            test_data = CROHME_DATA_PATH / 'test'
        else:
            test_data = MATHWRITING_DATA_PATH / 'test'
        logger.info(f"Using default test data directory: {test_data}")
    
    # Log command execution
    logger.info(f"Evaluating model from {model_path}")
    logger.info(f"Test data path: {test_data}")
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = DataLoader(
        data_dir=test_data.parent.parent,  # Get the parent of the 'test' directory
        batch_size=batch_size,
        input_size=(128, 128),  # Default size, should match model's input size
        use_augmentation=False  # No augmentation for evaluation
    )
    
    # Load test dataset
    logger.info(f"Loading {dataset} test dataset...")
    if dataset.lower() == 'crohme':
        test_ds = data_loader.load_crohme_dataset('test')
    else:  # mathwriting
        test_ds = data_loader.load_mathwriting_dataset('test')
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = MathTokenizer()
    
    # Initialize and load model
    logger.info("Loading model...")
    model_config = ModelConfig()
    model = MathRecognitionModel(
        model_config=model_config,
        tokenizer=tokenizer
    )
    
    try:
        model.load_weights(str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = model.evaluate(test_ds, verbose=1)
        
        # Log evaluation results
        logger.info(f"Test loss: {results[0]:.4f}")
        logger.info(f"Test accuracy: {results[1]:.4f}")
        
        # Save evaluation results if output directory is provided
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / "evaluation_results.txt"
            with open(output_file, 'w') as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Test data: {test_data}\n")
                f.write(f"Test loss: {results[0]:.4f}\n")
                f.write(f"Test accuracy: {results[1]:.4f}\n")
            logger.info(f"Evaluation results saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return None

@app.command()
def check_device():
    """Check available devices and compute capabilities."""
    # Try to configure Metal
    metal_configured = configure_metal()
    
    # Get and display device information
    device_info = get_compute_capability()
    
    # Print device information in a readable format
    logger.info("Device Information:")
    logger.info(f"Platform: {device_info['platform']}")
    logger.info(f"Processor: {device_info['processor']}")
    logger.info(f"Python version: {device_info['python_version']}")
    logger.info(f"TensorFlow version: {device_info['tensorflow_version']}")
    
    if device_info.get('has_metal', False):
        logger.info(f"TensorFlow-Metal version: {device_info.get('tensorflow_metal_version', 'Unknown')}")
    
    logger.info(f"Available devices: {', '.join(device_info['devices'])}")
    logger.info(f"GPU available: {device_info['has_gpu']}")
    logger.info(f"Metal available: {device_info['has_metal']}")
    
    # Additional information about Metal configuration
    logger.info(f"Metal GPU successfully configured: {metal_configured}")
    
    # Test a simple tensor operation
    if device_info['has_gpu']:
        logger.info("Testing GPU tensor operation...")
        with tf.device('/device:GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start_time = time.time()
            c = tf.matmul(a, b)
            elapsed = time.time() - start_time
            logger.info(f"GPU tensor operation completed in {elapsed:.4f} seconds")
    
    return device_info

if __name__ == "__main__":
    app()