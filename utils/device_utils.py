"""Utilities for device-specific configurations (CPU, GPU, NPU, etc.)."""

import logging
import os
import platform
import tensorflow as tf
from .logging_utils import get_logger

logger = get_logger(__name__)

def configure_metal():
    """Configure TensorFlow to use Apple Metal GPU acceleration if available.
    
    This function should be called early in the app initialization before 
    any TensorFlow operations are performed.
    
    Returns:
        bool: True if Metal GPU was successfully configured, False otherwise.
    """
    # Only attempt Metal configuration on macOS
    if platform.system() != "Darwin":
        logger.info("Not running on macOS, skipping Metal configuration")
        return False
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Metal GPU devices found: {gpus}")
        
        try:
            # Configure memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memory growth enabled for {gpu}")
            
            # Check if configuration was successful
            logger.info("Metal GPU acceleration is enabled")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring Metal GPU: {e}")
            return False
    else:
        logger.warning("No GPU devices found")
        return False

def get_compute_capability():
    """Get the compute capability of the current device.
    
    Returns:
        dict: Dictionary with device information.
    """
    # Get TensorFlow version 
    try:
        # Different ways to get TF version depending on the TF version
        tf_version = getattr(tf, "__version__", None)
        if tf_version is None:
            tf_version = getattr(tf.version, "VERSION", "Unknown")
    except:
        tf_version = "Unknown"
    
    device_info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf_version,
        "devices": [],
        "has_gpu": False,
        "has_metal": False,
        "has_npu": False,
    }
    
    # Check for Apple Silicon
    is_apple_silicon = (platform.system() == "Darwin" and platform.processor() == "arm")
    device_info["is_apple_silicon"] = is_apple_silicon
    
    # Get TensorFlow physical devices
    try:
        physical_devices = tf.config.list_physical_devices()
        device_info["devices"] = [str(device) for device in physical_devices]
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        device_info["has_gpu"] = len(gpu_devices) > 0
        
        # If we have GPU on macOS, assume it's Metal
        if device_info["has_gpu"] and is_apple_silicon:
            device_info["has_metal"] = True
            
    except Exception as e:
        logger.error(f"Error getting TensorFlow devices: {e}")
    
    return device_info

def set_training_memory_options(memory_limit=None, allow_growth=True):
    """Configure memory options for training.
    
    Args:
        memory_limit (int, optional): Memory limit in MB. None means no limit.
        allow_growth (bool): Whether to allow GPU memory growth.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        logger.info("No GPU found, memory options not applied")
        return
    
    try:
        # Apply options to all GPUs
        for gpu in gpus:
            # Allow memory growth
            if allow_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {gpu}")
            
            # Set memory limit if provided
            if memory_limit is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"GPU memory limit set to {memory_limit}MB for {gpu}")
                
    except RuntimeError as e:
        logger.error(f"Error configuring GPU memory options: {e}")

def get_optimal_batch_size(input_shape, base_batch_size=32):
    """Calculate an optimal batch size based on device memory.
    
    This is a simple heuristic for estimating batch size based on device type
    and input shape. For more precise batch size selection, you would need
    to benchmark with different batch sizes on your specific device.
    
    Args:
        input_shape (tuple): Shape of a single input item (H, W, C)
        base_batch_size (int): Base batch size for a standard GPU
        
    Returns:
        int: Recommended batch size
    """
    # Default batch size for CPU
    recommended_batch_size = base_batch_size
    
    # Check device type
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Metal GPUs might need different batch sizes based on 
        # available memory and compute capability
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Apple Silicon typically has unified memory, so can handle larger batches
            # But M1/M2/M3 devices have less dedicated GPU memory than high-end NVIDIA cards
            # Adjust based on your specific device performance
            ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
            
            # Scale batch size based on RAM (simple heuristic)
            if ram_gb > 16:  # Higher memory machines
                recommended_batch_size = base_batch_size * 2
            elif ram_gb > 8:  # Medium memory machines
                recommended_batch_size = base_batch_size
            else:  # Lower memory machines 
                recommended_batch_size = base_batch_size // 2
                
            logger.info(f"Detected Apple Silicon with ~{ram_gb:.1f}GB RAM, " 
                       f"recommended batch size: {recommended_batch_size}")
            
    # Additional adjustments based on input shape size
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    if input_size > 128 * 128 * 3:  # Larger than 128x128 RGB
        recommended_batch_size = max(4, recommended_batch_size // 2)
    
    return recommended_batch_size