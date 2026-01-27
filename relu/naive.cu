/**
Write a program to apply the ReLU (Rectified Linear Unit) activation function to an array of floats.
ReLU is one of the most widely used activation functions in deep learning and neural networks.

Problem:
Given an input array of N floating-point values, apply the ReLU function element-wise:
    ReLU(x) = max(0, x)
    
For each element:
    - If the value is positive (x >= 0), output the value unchanged
    - If the value is negative (x < 0), output 0

Example:
    input = [-2.5, 0.0, 3.7, -1.2, 5.0]
    output = [0.0, 0.0, 3.7, 0.0, 5.0]

Solution Approach:
Each thread processes one element of the input array independently. This implementation uses 
a bit manipulation optimization to check the sign of floating-point numbers rather than using 
a standard comparison operator. If the value is negative, the output is set to 0. Otherwise, 
the input value is copied to the output.

Optimization Tricks:
1. **Sign Bit Extraction via __float_as_uint(x) >> 31**:
   - IEEE 754 floats store the sign bit as the most significant bit (bit 31)
   - __float_as_uint() reinterprets the float's bits as an unsigned integer without conversion
   - Right shifting by 31 bits extracts just the sign bit: 0 for positive, 1 for negative
   - This avoids floating-point comparison overhead and can be faster than (x < 0)
   - Particularly effective on GPUs where integer operations can be cheaper than FP comparisons
   
2. **Early Return for Negative Values**:
   - When a negative value is detected, the kernel immediately returns after setting output to 0
   - This avoids executing the second conditional check, reducing divergence impact

Implementation Requirements:
- Use only native CUDA features (external libraries are not permitted)
- The solve function signature must remain unchanged
- All pointers (input, output) are device pointers to GPU memory
- The final result must be stored in the output array
 */

#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x<N && (__float_as_uint(input[x]) >> 31) != 0) {
        output[x] = 0.0f;
        return;
    }
    if (x<N){
        output[x] = input[x];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
