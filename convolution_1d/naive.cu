/**
Write a program to perform 1D convolution on an input array using a kernel (filter).
1D convolution is a fundamental operation in signal processing and deep learning where a kernel 
slides across an input array, computing the weighted sum at each position.

Problem:
Given an input array of size N and a kernel of size K, compute the convolution output.
At each valid position i in the input (where the kernel fits completely), compute:
    output[i] = sum(input[i+j] * kernel[j]) for j = 0 to K-1

The output size will be (N - K + 1), representing all positions where the kernel fits entirely 
within the input bounds (valid convolution, no padding).

Example:
    input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
    output[0] = 1*1 + 2*0 + 3*(-1) = -2
    output[1] = 2*1 + 3*0 + 4*(-1) = -2
    output[2] = 3*1 + 4*0 + 5*(-1) = -2
    result = [-2, -2, -2]

Solution Approach:
Each thread computes one output element by iterating through the kernel and accumulating 
the weighted sum of corresponding input values. This is a naive implementation where each 
thread performs O(K) operations independently.

Implementation Requirements:
- Use only native CUDA features (external libraries are not permitted)
- The solve function signature must remain unchanged
- All pointers (input, kernel, output) are device pointers to GPU memory
- The final convolution result must be stored in the output array
 */

#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < input_size - kernel_size+1) {
        float res = 0.0f;
        for (int j=0;j<kernel_size; ++j) {
            res += input[i+j]*kernel[j];
        }
        output[i] = res;
    }

}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
