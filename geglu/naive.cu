/**
Write a program to apply the GEGLU (Gated Gaussian Error Linear Unit) activation function.
GEGLU is an advanced activation function used in modern transformer architectures and neural
networks, particularly effective in feed-forward layers. It combines gating mechanisms with
the smooth, non-linear properties of GELU.

Problem:
Given an input array of N floats, split it into two halves and compute GEGLU:
    GEGLU(x, x') = x * GELU(x')
    
where:
    - x comes from the first half: input[0] to input[N/2-1]
    - x' comes from the second half: input[N/2] to input[N-1]
    - GELU(x') = 0.5 * x' * (1 + erf(x' / sqrt(2)))
    - erf() is the Gaussian error function
    
The output array has size N/2, where each element is the product of a value from the first
half and the GELU of the corresponding value from the second half.

Mathematical Formula:
    For each i in [0, N/2):
        x = input[i]
        x' = input[i + N/2]
        GELU(x') = 0.5 * x' * (1 + erf(x' * (1/sqrt(2))))
        output[i] = x * GELU(x')

Example:
    N = 6
    input = [2.0, 1.0, -1.0,  |  3.0, 0.5, -0.5]
            ← first half x →   ← second half x' →
    
    For i=0: x=2.0, x'=3.0
        GELU(3.0) ≈ 0.5 * 3.0 * (1 + erf(3.0/√2)) ≈ 2.996
        output[0] = 2.0 * 2.996 ≈ 5.992
    
    For i=1: x=1.0, x'=0.5
        GELU(0.5) ≈ 0.5 * 0.5 * (1 + erf(0.5/√2)) ≈ 0.346
        output[1] = 1.0 * 0.346 ≈ 0.346
    
    (output has size 3: [5.992, 0.346, ...])

Solution Approach:
Each thread processes one pair of elements (one from each half of the input). The thread
computes GELU for the second half element, then multiplies it with the first half element.
This is an embarrassingly parallel operation where threads work completely independently.

Optimization Tricks:
1. **Precomputed Mathematical Constant**:
   - constexpr float inv_sqrt2 = 1/sqrt(2) ≈ 0.7071067811865475
   - Computed at compile time, not runtime
   - Eliminates the need for division or sqrt() in the kernel
   - Each thread saves one expensive floating-point division operation

2. **Input Splitting Strategy**:
   - Input array is logically split: first half (x) and second half (x')
   - Index arithmetic: thread i reads input[i] and input[i + halfN]
   - No need to physically copy or rearrange data
   - Memory accesses are coalesced for both halves

3. **Built-in Error Function**:
   - Uses CUDA's erff() function (single-precision error function)
   - Hardware-accelerated on modern GPUs using special function units (SFUs)
   - Much faster than implementing erf approximation manually
   - Provides good accuracy while maintaining performance

4. **Reduced Output Size**:
   - Output array is half the size of input (N/2 instead of N)
   - Each thread writes exactly once to a unique output location
   - No write conflicts or synchronization needed
   - Better memory efficiency for subsequent operations in the pipeline

5. **Expression Evaluation Order**:
   - GELU computed first and stored in local variable
   - Reuses gelu value to compute final result
   - Compiler can optimize register usage and instruction scheduling
   - Clear, maintainable code that matches mathematical notation

Implementation Requirements:
- Use only native CUDA features (external libraries are not permitted)
- The solve function signature must remain unchanged
- All pointers (input, output) are device pointers to GPU memory
- Input array must have even size N (N/2 must be an integer)
- Output array has size N/2
 */

#include <cuda_runtime.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x<halfN) {
        float x2 = input[x+halfN];
        float gelu = (0.5f * x2) * (1 + erff(x2*inv_sqrt2));
        float geglu = input[x]*gelu;
        output[x] = geglu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
