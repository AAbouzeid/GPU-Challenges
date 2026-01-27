/**
Write a program to count the number of elements in a 2D matrix that equal a specific value K.
This extends the 1D counting problem to 2D arrays, commonly used in image processing, 
matrix analysis, and deep learning operations where counting specific values in feature maps
or activation matrices is needed.

Problem:
Given a 2D matrix (stored as a 1D array in row-major order) of size N×M and a target value K,
count how many elements in the matrix equal K. Store the final count in the output pointer.

The 2D matrix is flattened in row-major order: element at (row, col) is at index row*M + col.

Example:
    matrix = [[1, 3, 5],     N = 3, M = 3, K = 3
              [3, 7, 3],
              [9, 3, 2]]
    output = 4  (four elements equal 3)
    
    matrix = [[2, 4],        N = 2, M = 2, K = 1
              [6, 8]]
    output = 0  (no elements equal 1)

Solution Approach:
This implementation uses a 2D thread block layout to match the 2D structure of the input matrix.
Each thread checks one matrix element. Threads within each block use atomic operations to
accumulate matches into a shared memory counter. Finally, one thread per block adds the block's
total to the global output. This is the 2D extension of the naive 1D atomic approach.

Optimization Tricks:
1. **2D Thread Block Organization**:
   - Uses dim3 with 16×16 threads per block (256 threads total)
   - Natural mapping: each thread (threadIdx.x, threadIdx.y) corresponds to a matrix element
   - Simplifies index calculation: row = blockIdx.y * blockDim.y + threadIdx.y
   - Better code readability and maintainability for 2D data structures

2. **Two-Level Atomic Reduction**:
   - Level 1: Threads within a block use atomicAdd to shared memory (blockSum)
   - Level 2: One thread per block (thread 0,0) uses atomicAdd to global memory
   - Reduces global memory atomic contention from (N×M) to (gridSize) operations
   - For a 1024×1024 matrix with 16×16 blocks: reduces from ~1M to ~4K global atomics

3. **Shared Memory for Block-Level Aggregation**:
   - Single __shared__ int blockSum stores the count for each 2D thread block
   - Initialized by thread (0,0) to avoid redundant initialization
   - All threads in the 2D block contribute to this shared counter

4. **Thread Synchronization Points**:
   - First __syncthreads() ensures blockSum is initialized before any thread writes to it
   - Second __syncthreads() ensures all threads have finished counting before final atomic add
   - Critical for correctness in 2D thread block coordination

5. **Row-Major Memory Layout**:
   - Matrix stored contiguously in memory: row*M + col indexing
   - Threads in the same row (consecutive threadIdx.x) access consecutive memory locations
   - Enables memory coalescing when threads read matrix elements simultaneously

Implementation Requirements:
- Use only native CUDA features (external libraries are not permitted)
- The solve function signature must remain unchanged
- All pointers (input, output) are device pointers to GPU memory
- Input matrix is stored in row-major order as a 1D array
- The output pointer should point to a single integer that will contain the final count
- Assume output is pre-initialized to 0 before calling solve
 */

#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    // Initialize the block sum to zero
    __shared__ int blockSum;
    if (threadIdx.x == 0 && threadIdx.y == 0) blockSum = 0;
    __syncthreads();

    // Block sum if row, col in range
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row<N && col<M) {
        if (input[row*M+col] == K) {
        atomicAdd(&blockSum, 1);
    }
    }
    __syncthreads();

    // final output
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(output, blockSum);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}
