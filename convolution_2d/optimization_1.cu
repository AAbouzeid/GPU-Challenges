#include <cuda_runtime.h>

// Max kernel size: 31x31 = 961 elements (~3.8KB, well under 64KB limit)
#define MAX_KERNEL_SIZE 961

__constant__ float d_kernel[MAX_KERNEL_SIZE];

__global__ void convolution_2d_kernel(const float* input, float* output,
                                      int input_rows, int input_cols, 
                                      int kernel_rows, int kernel_cols) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i < input_cols - kernel_cols + 1 && j < input_rows - kernel_rows + 1) {
        float res = 0.0f;
        for (int k = 0; k < kernel_rows; ++k) {
            for (int l = 0; l < kernel_cols; ++l) {
                res += input[(j + k) * input_cols + i + l] * d_kernel[k * kernel_cols + l];
            }
        }
        int output_cols = input_cols - kernel_cols + 1;
        output[j * output_cols + i] = res;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, 
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    // Copy kernel to constant memory (device-to-device since kernel is already on device)
    cudaMemcpyToSymbol(d_kernel, kernel, kernel_rows * kernel_cols * sizeof(float), 
                       0, cudaMemcpyDeviceToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((output_cols + 15) / 16, (output_rows + 15) / 16);

    convolution_2d_kernel<<<gridDim, blockDim>>>(input, output, input_rows, input_cols, 
                                                  kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
 