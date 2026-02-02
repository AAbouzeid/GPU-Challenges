#include <cuda_runtime.h>

 __global__ void histogram(const int* input, int* histogram, int N, int num_bins) 
 {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // shared memory for frequency of each bin
    extern __shared__ int freq[];   // Dynamic shared memory
    for (int j = threadIdx.x; j < num_bins; j += blockDim.x) {
        freq[j] = 0;
    }
    
    __syncthreads();

    // count frequency of each bin
    if (i < N) {
        atomicAdd(&freq[input[i]], 1);
    }
    __syncthreads();

    // add frequency of each bin to histogram
    for (int j = threadIdx.x; j < num_bins; j += blockDim.x) {
        atomicAdd(&histogram[j], freq[j]); 
    }
    __syncthreads();
 }

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedMemSize = num_bins * sizeof(int);
    histogram<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
