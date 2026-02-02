#include <cuda_runtime.h>
#include <cmath>

__global__ void block_max_kernel(const float* input, float* blockMax, int N) {
    int id =  threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Calculate the maximum per thread over stride distance
    float m = -INFINITY;
    for (int i=tid; i<N; i+=stride) {
        m = fmaxf(m, __ldg(&input[i]));
    }

    // store it in s[tid] and sync
    __shared__ float s[256];
    s[threadIdx.x] = m;
    __syncthreads();

    // Reduce the s array
    for (int off = 128; off > 0; off >>= 1){
        if (id < off) s[id] = fmaxf(s[id], s[id + off]);
        __syncthreads();
    }

    // return max
    if (id == 0) blockMax[blockIdx.x] = s[0];
}

__global__ void global_max(const float* blockMax, float* outMax, int blocksPerGrid) {
    int tid = threadIdx.x;
    float m = -INFINITY;

    for (int i = tid; i<blocksPerGrid; i+=blockDim.x){
        m = fmaxf(m, __ldg(&blockMax[i]));
    }

    __shared__ float s[256];
    s[tid] = m;
    __syncthreads();

    // Reduce the blockMax array
    for (int off = 128; off > 0; off >>= 1){
        if (tid < off) s[tid] = fmaxf(s[tid], s[tid + off]);
        __syncthreads();
    }

    if (tid==0) outMax[0] = s[0];    
}

__global__ void exp_and_sum(const float* input, float* output, float* arrSum, int N, const float* max) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local = 0.0f;
    float max_val = __ldg(&max[0]);
    
    // Compute exp, write to output, and accumulate sum
    for (int i = x; i < N; i += stride) {
        float exp_val = __expf(__ldg(&input[i]) - max_val);
        output[i] = exp_val;  // Write exp values to output
        local += exp_val;
    }
    
    __shared__ float s[256];
    s[threadIdx.x] = local;
    __syncthreads();

    for (int off = 128; off > 0; off >>=1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x+off];
        __syncthreads();
    }

    if (threadIdx.x==0) atomicAdd(arrSum, s[0]);
}


__global__ void softmax_kernel(float* output, const float* sum, int N) {
    __shared__ float inv;
    if (threadIdx.x == 0) inv = __frcp_rn(__ldg(&sum[0]));
    __syncthreads();

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x<N) {
        output[x] *= inv;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 1.Compute the max for each block
    float* d_blockMax;
    cudaMalloc(&d_blockMax, blocksPerGrid * sizeof(float));
    block_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_blockMax, N);

    // 2. Reduce the maximum of the array
    float* d_globMax;
    cudaMalloc(&d_globMax, sizeof(float));
    global_max<<<1, threadsPerBlock>>>(d_blockMax, d_globMax, blocksPerGrid);

    // 3. Fused: compute exp, write output, and sum in one pass
    float* arr_sum;
    cudaMalloc(&arr_sum, sizeof(float));
    cudaMemset(arr_sum, 0, sizeof(float));
    exp_and_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, arr_sum, N, d_globMax);

    // 4. Normalize by sum
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, arr_sum, N);


    cudaFree(d_blockMax);
    cudaFree(d_globMax);
    cudaFree(arr_sum);
    cudaDeviceSynchronize();
}

