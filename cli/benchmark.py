"""Benchmark script for model performance testing."""

import argparse
import numpy as np
import tensorflow as tf
import time
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model.recognition import MathRecognitionModel
from utils.device_utils import configure_metal, get_compute_capability
from config import DeviceConfig

def benchmark_model(model_path, num_runs=100, batch_size=1, input_shape=(128, 128, 1), 
                    quantized=False, use_metal=True, log_level="INFO"):
    """Benchmark model inference performance.
    
    Args:
        model_path: Path to saved model
        num_runs: Number of inference runs to average
        batch_size: Batch size for inference
        input_shape: Input tensor shape
        quantized: Whether to use quantized model
        use_metal: Whether to use Metal GPU on Mac
        log_level: Logging level
    
    Returns:
        Dictionary with benchmark results
    """
    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level))
    logger = logging.getLogger(__name__)
    
    # Configure Metal if requested
    if use_metal:
        configure_metal()
    
    # Get device info
    device_info = get_compute_capability()
    logger.info(f"Running benchmark on: {device_info}")
    
    # Create model
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model
        if quantized:
            # For TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create random input data
            input_shape = input_details[0]['shape']
            logger.info(f"Model input shape: {input_shape}")
            input_data = np.random.random_sample(input_shape).astype(np.float32)
            
            # Warm-up run
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Benchmark runs
            logger.info(f"Running {num_runs} inference passes...")
            times = []
            for i in range(num_runs):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                end_time = time.time()
                times.append(end_time - start_time)
                
                if i % 10 == 0:
                    logger.info(f"Completed {i+1}/{num_runs} runs")
        else:
            # For regular TensorFlow model
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded: {model.summary()}")
            
            # Create random input data
            if isinstance(input_shape, tuple) and len(input_shape) == 3:
                # Add batch dimension if not present
                input_shape = (batch_size,) + input_shape
            
            logger.info(f"Input shape for benchmark: {input_shape}")
            input_data = np.random.random_sample(input_shape).astype(np.float32)
            
            # Warm-up run
            _ = model.predict(input_data, verbose=0)
            
            # Benchmark runs
            logger.info(f"Running {num_runs} inference passes...")
            times = []
            for i in range(num_runs):
                start_time = time.time()
                _ = model.predict(input_data, verbose=0)
                end_time = time.time()
                times.append(end_time - start_time)
                
                if i % 10 == 0:
                    logger.info(f"Completed {i+1}/{num_runs} runs")
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Get device config for comparison with target
        device_config = DeviceConfig()
        meets_target = avg_time * 1000 <= device_config.target_latency_ms
        
        # Print results
        logger.info(f"\nBenchmark Results:")
        logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
        logger.info(f"Standard deviation: {std_time*1000:.2f} ms")
        logger.info(f"Min inference time: {min_time*1000:.2f} ms")
        logger.info(f"Max inference time: {max_time*1000:.2f} ms")
        logger.info(f"Target latency: {device_config.target_latency_ms} ms")
        logger.info(f"Meets target latency: {meets_target}")
        
        results = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "target_latency_ms": device_config.target_latency_ms,
            "meets_target": meets_target,
            "device_info": device_info,
            "model_path": str(model_path),
            "quantized": quantized
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference performance")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of inference runs to average")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--quantized", action="store_true", help="Whether the model is quantized (TFLite)")
    parser.add_argument("--use_metal", action="store_true", help="Use Metal GPU acceleration on Mac")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    benchmark_model(
        model_path=args.model_path,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        quantized=args.quantized,
        use_metal=args.use_metal,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()