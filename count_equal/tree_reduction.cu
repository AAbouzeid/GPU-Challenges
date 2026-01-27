/**
Write a program to count the number of elements in an array that equal a specific value K.
This implementation uses tree reduction for optimal performance by avoiding atomic operations
within thread blocks.

Problem:
Given an input array of N integers and a target value K, count how many elements in the array
equal K. Store the final count in the output pointer.

Example:
    input = [1, 3, 5, 3, 7, 3, 9], K = 3
    output = 3  (three elements equal 3)
    
    input = [2, 4, 6, 8], K = 1
    output = 0  (no elements equal 1)

Solution Approach:
This implementation uses a tree reduction strategy within each block to efficiently aggregate
counts without atomic operations. Each thread first checks its element and stores a 1 or 0
in shared memory. Then threads cooperatively perform a parallel tree reduction, where pairs
of elements are summed in each iteration until a single sum remains. Finally, one thread per
block adds the block's total to the global output.

Optimization Tricks:
1. **Tree Reduction for Block-Level Aggregation**:
   - Instead of all threads using atomicAdd to shared memory, use tree reduction
   - In each iteration, half the active threads sum pairs: sdata[i] += sdata[i + stride]
   - Number of iterations: log2(blockSize), providing O(log n) time complexity per block
   - Example with 8 threads: [1,0,1,1,0,1,0,1] → [1,2,1,1] → [3,2] → [5]
   - Eliminates atomic contention within blocks completely

2. **Memory Coalescing in Initial Load**:
   - Each thread directly loads its element and evaluates the condition
   - Initial shared memory write is coalesced: consecutive threads write consecutive addresses
   - Better memory bandwidth utilization compared to scattered atomic updates

3. **Reduced Global Atomic Contention**:
   - Only one atomic operation per block to global memory
   - For a problem with millions of elements, reduces global atomics from millions to thousands
   - Global atomic becomes the only serialization point, not a per-thread bottleneck

4. **Better GPU Occupancy**:
   - Tree reduction has no atomic serialization within blocks
   - All threads in a block can execute their reduction steps in parallel
   - Improves instruction throughput and latency hiding

Performance Comparison:
- Naive atomic approach: O(N) atomic operations in shared memory + O(gridSize) global atomics
- Tree reduction: O(N) + O(N log blockSize) in parallel + O(gridSize) global atomics
- For large blocks (256+ threads), tree reduction significantly outperforms atomics
- Typical speedup: 2-5x faster than atomic-based reduction for large arrays

Implementation Requirements:
- Use only native CUDA features (external libraries are not permitted)
- The solve function signature must remain unchanged
- All pointers (input, output) are device pointers to GPU memory
- The output pointer should point to a single integer that will contain the final count
- Assume output is pre-initialized to 0 before calling solve

ONLY OUTPERFORMS THE NAIVE APPROACH IF a lot of elements are equal to K.
 */

#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    __shared__ int sdata[256];  // Shared memory for reduction (sized for max threads per block)
    
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Load element and check condition
    sdata[tid] = (i < N && input[i] == K) ? 1 : 0;
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 adds block's result to global output
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
