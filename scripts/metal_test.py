"""
Metal GPU verification test for TensorFlow.
Trains a small model on CIFAR-100 to verify Metal acceleration is working.
"""

import tensorflow as tf
import time
import platform

print(f"Python version: {platform.python_version()}")
print(f"TensorFlow version: {tf.version.VERSION}")

try:
    # Try to import tensorflow_metal
    import tensorflow_metal
    print(f"TensorFlow Metal plugin is available")
except ImportError:
    print("TensorFlow Metal plugin is NOT available")

# List available devices
physical_devices = tf.config.list_physical_devices()
print(f"Physical devices: {physical_devices}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Check if Metal plugin is enabled
if tf.config.list_physical_devices('GPU'):
    print("Metal GPU acceleration is ENABLED")
else:
    print("Metal GPU acceleration is NOT enabled")

# Load a small dataset for testing
print("Loading CIFAR-100 dataset...")
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Use a smaller subset for quick testing
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# Create a simple model
print("Creating model...")
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,
)

# Configure the model
print("Compiling model...")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Time the training to see if GPU acceleration is working
print("Training model (this will take some time)...")
start_time = time.time()

# Train for just 1 epoch for quick testing
history = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

print(f"Training completed in {training_time:.2f} seconds")
print("Metal GPU support verification test completed")