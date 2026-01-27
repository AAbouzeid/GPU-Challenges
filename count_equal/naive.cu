/**
Write a program to count the number of elements in an array that equal a specific value K.
This is a fundamental parallel reduction problem that appears in filtering, histogram computation,
and conditional aggregation operations in data processing and machine learning.

Problem:
Given an input array of N integers and a target value K, count how many elements in the array
equal K. Store the final count in the output pointer.

Example:
    input = [1, 3, 5, 3, 7, 3, 9], K = 3
    output = 3  (three elements equal 3)
    
    input = [2, 4, 6, 8], K = 1
    output = 0  (no elements equal 1)

Solution Approach:
This implementation uses a two-level reduction strategy to efficiently aggregate counts across
threads. First, threads within each block count matching elements into shared memory. Then,
one thread per block adds the block's total count to the global output.

Optimization Tricks:
1. **Two-Level Atomic Reduction**:
   - Level 1: Threads within a block use atomicAdd to accumulate into shared memory (block_count)
   - Level 2: One thread per block uses atomicAdd to accumulate into global memory (output)
   - This dramatically reduces global memory atomic contention compared to all threads using global atomics
   - Shared memory atomics are much faster than global memory atomics on most GPU architectures

2. **Shared Memory for Block-Level Aggregation**:
   - __shared__ int block_count creates one counter per thread block in fast shared memory
   - Initialized to 0 by thread 0, avoiding redundant initialization by all threads
   - All threads in the block contribute to this shared counter in parallel

3. **Thread Synchronization with __syncthreads()**:
   - First __syncthreads() ensures block_count is initialized before any thread uses it
   - Second __syncthreads() ensures all threads have finished counting before the final atomic add
   - Critical for correctness in multi-threaded block execution

4. **Single-Thread Global Update**:
   - Only threadIdx.x == 0 performs the global atomic add
   - Reduces global atomic operations from threadsPerBlock to 1 per block
   - For 256 threads/block, this is a 256x reduction in global atomic contention

Implementation Requirements:
- Use only native CUDA features (external libraries are not permitted)
- The solve function signature must remain unchanged
- All pointers (input, output) are device pointers to GPU memory
- The output pointer should point to a single integer that will contain the final count
- Assume output is pre-initialized to 0 before calling solve
 */

#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    __shared__ int block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N && input[i] == K) atomicAdd(&block_count, 1);

    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(output, block_count);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
