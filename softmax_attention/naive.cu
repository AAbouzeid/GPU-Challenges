#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // row in input
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // col in input
    if (r < rows && c < cols) {
        output[c * rows + r] = input[r * cols + c];
    }
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Shared memory tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Local accumulator (in registers - fast!)
    float sum = 0.0f;

    // Loop over all tiles along the shared dimension N
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A: each thread loads one element
        // A[row][t*TILE_SIZE + threadIdx.x]
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;  // Pad with zeros for out-of-bounds
        }

        // Load tile from B: each thread loads one element
        // B[t*TILE_SIZE + threadIdx.y][col]
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < N && col < K) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * K + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // Wait before loading next tile (so we don't overwrite data still in use)
        __syncthreads();
    }

    // Write final result to global memory
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// Row-wise softmax with fused scaling: each block handles one row
__global__ void row_softmax_kernel(float* matrix, int M, int N, float scale) {
    int row = blockIdx.x;  // Each block handles one row
    int tid = threadIdx.x;
    
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    
    // Step 1: Find max in this row (with scaling applied)
    float local_max = -INFINITY;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, matrix[row * N + col] * scale);
    }
    s_max[tid] = local_max;
    __syncthreads();
    
    // Tree reduction for max (reused pattern from optimized_1.cu)
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s_max[tid] = fmaxf(s_max[tid], s_max[tid + off]);
        __syncthreads();
    }
    float row_max = s_max[0];
    __syncthreads();
    
    // Step 2: Compute exp(x*scale - max) and sum (fused, like optimized_1.cu)
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float exp_val = __expf(matrix[row * N + col] * scale - row_max);
        matrix[row * N + col] = exp_val;  // Store exp values back
        local_sum += exp_val;
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    
    // Tree reduction for sum
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s_sum[tid] += s_sum[tid + off];
        __syncthreads();
    }
    float row_sum = s_sum[0];
    __syncthreads();
    
    // Step 3: Normalize by sum (like softmax_kernel in optimized_1.cu)
    float inv_sum = __frcp_rn(row_sum);  // Fast reciprocal
    for (int col = tid; col < N; col += blockDim.x) {
        matrix[row * N + col] *= inv_sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) 
{
    // 0. Setup - 2D blocks for transpose and matmul
    dim3 blockDim2D(TILE_SIZE, TILE_SIZE);

    // 1. Compute the transpose of K
    float* d_kTrans;
    cudaMalloc(&d_kTrans, N * d * sizeof(float));
    dim3 transposeGrid((d + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matrix_transpose_kernel<<<transposeGrid, blockDim2D>>>(K, d_kTrans, N, d);

    // 2. Compute Q × K^T = scores: Q(M×d) × K^T(d×N) → scores(M×N)
    float* d_scores;
    cudaMalloc(&d_scores, M * N * sizeof(float));
    dim3 matmulGrid1((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrix_multiplication_kernel<<<matmulGrid1, blockDim2D>>>(Q, d_kTrans, d_scores, M, d, N);

    // 3 & 4. Fused: Scale by 1/sqrt(d) + Row-wise softmax (1D launch, M blocks)
    row_softmax_kernel<<<M, 256>>>(d_scores, M, N, 1.0f / sqrtf((float)d));

    // 5. Multiply scores × V: scores(M×N) × V(N×d) → output(M×d)
    dim3 matmulGrid2((d + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrix_multiplication_kernel<<<matmulGrid2, blockDim2D>>>(d_scores, V, output, M, N, d);

    // Cleanup
    cudaFree(d_kTrans);
    cudaFree(d_scores);
    cudaDeviceSynchronize();
}
