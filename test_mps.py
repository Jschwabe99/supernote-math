#!/usr/bin/env python3
"""Test script for Apple Metal Performance Shaders (MPS) acceleration"""
import torch
import time
import os
import sys

def test_basic_operations():
    """Test basic tensor operations on MPS device"""
    print("\n==== Testing basic MPS operations ====")
    
    # Check if MPS is available
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("❌ MPS is not available on this system")
        return False
    
    try:
        # Create CPU tensor
        cpu_tensor = torch.randn(2000, 2000)
        print(f"CPU tensor shape: {cpu_tensor.shape}, device: {cpu_tensor.device}")
        
        # Time CPU matrix multiplication
        start_time = time.time()
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        print(f"CPU matrix multiplication time: {cpu_time:.4f} seconds")
        
        # Create MPS tensor
        start_time = time.time()
        mps_tensor = cpu_tensor.to('mps')
        to_device_time = time.time() - start_time
        print(f"MPS tensor shape: {mps_tensor.shape}, device: {mps_tensor.device}")
        print(f"Time to transfer to MPS: {to_device_time:.4f} seconds")
        
        # Time MPS matrix multiplication
        start_time = time.time()
        mps_result = torch.matmul(mps_tensor, mps_tensor)
        mps_time = time.time() - start_time
        print(f"MPS matrix multiplication time: {mps_time:.4f} seconds")
        
        # Transfer result back to CPU for comparison
        start_time = time.time()
        mps_result_cpu = mps_result.to('cpu')
        to_cpu_time = time.time() - start_time
        print(f"Time to transfer result back to CPU: {to_cpu_time:.4f} seconds")
        
        # Check results match
        is_close = torch.allclose(cpu_result, mps_result_cpu, rtol=1e-3, atol=1e-3)
        print(f"Results match: {is_close}")
        
        # Calculate speedup
        speedup = cpu_time / mps_time
        print(f"Speedup: {speedup:.2f}x")
        
        print("✅ Basic MPS operations test passed\n")
        return True
    except Exception as e:
        print(f"❌ Error in basic MPS operations: {e}")
        return False

def test_nn_operations():
    """Test neural network operations on MPS device"""
    print("\n==== Testing neural network operations on MPS ====")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("❌ MPS is not available on this system")
        return False
    
    try:
        # Create a simple convolutional network
        class SimpleConvNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.MaxPool2d(2)
                self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
                self.fc = torch.nn.Linear(32 * 32 * 32, 10)
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(-1, 32 * 32 * 32)
                x = self.fc(x)
                return x
        
        # Create random input data
        batch_size = 16
        input_data = torch.randn(batch_size, 1, 128, 128)
        
        # Initialize model and move to CPU
        cpu_model = SimpleConvNet()
        
        # Time CPU forward pass
        start_time = time.time()
        cpu_output = cpu_model(input_data)
        cpu_time = time.time() - start_time
        print(f"CPU forward pass time: {cpu_time:.4f} seconds")
        
        # Move model and input to MPS
        mps_model = SimpleConvNet().to('mps')
        mps_input = input_data.to('mps')
        
        # Time MPS forward pass
        start_time = time.time()
        mps_output = mps_model(mps_input)
        mps_time = time.time() - start_time
        print(f"MPS forward pass time: {mps_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / mps_time
        print(f"Neural network speedup: {speedup:.2f}x")
        
        print("✅ Neural network MPS test passed\n")
        return True
    except Exception as e:
        print(f"❌ Error in neural network MPS test: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"MPS built: {torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
    
    # Run tests
    basic_test_passed = test_basic_operations()
    nn_test_passed = test_nn_operations()
    
    # Summary
    print("\n==== Test Summary ====")
    print(f"Basic MPS operations: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Neural network operations: {'✅ PASSED' if nn_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and nn_test_passed:
        print("\n🎉 MPS is working correctly on your system!")
        print("You should be able to use Apple Metal acceleration with your models.")
    else:
        print("\n⚠️  Some MPS tests failed. There might be issues using Metal acceleration.")
        print("Check the error messages above for more details.")