# Use Python 3.7 slim-buster as the base image
FROM python:3.7-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install essential system dependencies needed by OpenCV, PyTorch, etc.
# Combine update, install, and cleanup in one RUN command for fewer layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# --- CRITICAL STEP for PyTorch 1.8.1 ---
# Install a compatible NumPy version BEFORE PyTorch to avoid C-API conflicts.
RUN pip install --no-cache-dir numpy==1.19.5

# Install PyTorch, torchvision, and other specific versions using the stable URL
# This ensures correct wheels are fetched, especially for older versions.
RUN pip install --no-cache-dir \
    torch==1.8.1 \
    torchvision==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install the rest of the Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
# This is done *after* installing dependencies to leverage Docker build cache
COPY . .

# Set environment variables (if PosFormer code is mounted separately)
# Assumes PosFormer code will be mounted at /posformer inside the container
ENV PYTHONPATH="/app:/posformer:${PYTHONPATH}"

# Default command to run when the container starts (provides an interactive shell)
CMD ["/bin/bash"]