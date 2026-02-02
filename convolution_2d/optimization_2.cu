#include <cuda_runtime.h>

// Max kernel size: 31x31 = 961 elements
#define MAX_KERNEL_SIZE 961
#define BLOCK_SIZE 16
#define MAX_KERNEL_DIM 31

__constant__ float d_kernel[MAX_KERNEL_SIZE];

__global__ void convolution_2d_kernel(const float* input, float* output,
                                      int input_rows, int input_cols, 
                                      int kernel_rows, int kernel_cols) 
{
    // Shared memory tile: BLOCK_SIZE outputs + (kernel_dim - 1) halo on each axis
    // Max size: (16 + 30) x (16 + 30) = 46 x 46 = 2116 floats = ~8.5KB
    __shared__ float tile[BLOCK_SIZE + MAX_KERNEL_DIM - 1][BLOCK_SIZE + MAX_KERNEL_DIM - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global output coordinates this thread computes
    int out_x = blockIdx.x * BLOCK_SIZE + tx;
    int out_y = blockIdx.y * BLOCK_SIZE + ty;
    
    int output_cols = input_cols - kernel_cols + 1;
    int output_rows = input_rows - kernel_rows + 1;
    
    // Tile dimensions needed: BLOCK_SIZE + kernel_size - 1
    int tile_width = BLOCK_SIZE + kernel_cols - 1;
    int tile_height = BLOCK_SIZE + kernel_rows - 1;
    
    // Top-left corner of this block's input tile in global memory
    int tile_start_x = blockIdx.x * BLOCK_SIZE;
    int tile_start_y = blockIdx.y * BLOCK_SIZE;
    
    // Cooperatively load the tile into shared memory
    // Each thread may need to load multiple elements since tile > block
    for (int load_y = ty; load_y < tile_height; load_y += BLOCK_SIZE) {
        for (int load_x = tx; load_x < tile_width; load_x += BLOCK_SIZE) {
            int global_x = tile_start_x + load_x;
            int global_y = tile_start_y + load_y;
            
            // Bounds check for input
            if (global_x < input_cols && global_y < input_rows) {
                tile[load_y][load_x] = input[global_y * input_cols + global_x];
            } else {
                tile[load_y][load_x] = 0.0f;
            }
        }
    }
    
    // Wait for all threads to finish loading
    __syncthreads();
    
    // Compute convolution using shared memory
    if (out_x < output_cols && out_y < output_rows) {
        float res = 0.0f;
        for (int k = 0; k < kernel_rows; ++k) {
            for (int l = 0; l < kernel_cols; ++l) {
                res += tile[ty + k][tx + l] * d_kernel[k * kernel_cols + l];
            }
        }
        output[out_y * output_cols + out_x] = res;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, 
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, kernel, kernel_rows * kernel_cols * sizeof(float), 
                       0, cudaMemcpyDeviceToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (output_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolution_2d_kernel<<<gridDim, blockDim>>>(input, output, input_rows, input_cols, 
                                                  kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
 