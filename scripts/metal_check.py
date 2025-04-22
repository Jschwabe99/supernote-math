"""
Simple check for Metal GPU support in TensorFlow.
"""

import sys
import platform
import pkg_resources

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

# Check installed packages
print("\nInstalled packages:")
packages = ['tensorflow', 'tensorflow-metal', 'tensorflow-macos']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"  {package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"  {package}: Not installed")

# Try to import TensorFlow
print("\nTesting TensorFlow import:")
try:
    import tensorflow as tf
    print("  TensorFlow imported successfully")
except ImportError as e:
    print(f"  Error importing TensorFlow: {e}")
    sys.exit(1)

# Try to import TensorFlow Metal
print("\nTesting tensorflow-metal import:")
try:
    import tensorflow_metal
    print("  TensorFlow Metal plugin imported successfully")
except ImportError as e:
    print(f"  Error importing TensorFlow Metal: {e}")

# Check physical devices
print("\nDetecting devices:")
try:
    physical_devices = tf.config.list_physical_devices()
    print(f"  All physical devices: {physical_devices}")
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"  GPU devices: {gpu_devices}")
    
    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"  CPU devices: {cpu_devices}")
    
    if gpu_devices:
        print("  ✓ Metal GPU acceleration is ENABLED")
    else:
        print("  ✗ Metal GPU acceleration is NOT enabled")
        
except Exception as e:
    print(f"  Error detecting devices: {e}")

# Try simple TF operations
print("\nRunning simple TensorFlow operation:")
try:
    # Create a simple tensor and perform a GPU operation
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        result = tf.reduce_sum(c)
        
    print(f"  Operation result: {result.numpy()}")
    print("  ✓ TensorFlow operations completed successfully")
except Exception as e:
    print(f"  Error running TensorFlow operations: {e}")
    
print("\nMetal GPU support check completed")