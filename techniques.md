# CUDA Optimization Techniques

## 1. Two-Level Atomic Reduction
**Example:** `count_equal/naive.cu`, `count_equal_2d/naive.cu`

Reduces atomic contention by using a hierarchy of atomic operations:
- **Level 1:** Threads within a block use `atomicAdd()` to a `__shared__` variable
- **Level 2:** One thread per block uses `atomicAdd()` to global memory

```cuda
__shared__ int block_count;
if (threadIdx.x == 0) block_count = 0;
__syncthreads();

// Each thread atomically updates shared memory
if (condition) atomicAdd(&block_count, 1);

__syncthreads();
// One thread updates global memory
if (threadIdx.x == 0) atomicAdd(output, block_count);
```

**Benefit:** Reduces global atomic operations from O(N) to O(grid_size), dramatically lowering memory contention.

---

## 2. Tree Reduction
**Example:** `count_equal/tree_reduction.cu`, `tensor_reduction_1d/tree_reduction.cu`

Eliminates atomic operations within blocks by using parallel pairwise summation:

```cuda
__shared__ float sdata[256];
sdata[tid] = local_value;
__syncthreads();

// Parallel reduction: stride halves each iteration
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}

if (tid == 0) atomicAdd(output, sdata[0]);
```

**Benefit:** O(log n) time complexity per block with no intra-block atomic contention. Typically 2-5x faster than atomic-based reduction for large blocks.

---

## 3. Grid-Stride Loop
**Example:** `tensor_reduction_1d/tree_reduction.cu`

Allows each thread to process multiple elements, handling arrays larger than grid size:

```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

float local = 0.0f;
for (int i = tid; i < N; i += stride) {
    local += input[i];
}
```

**Benefit:** Handles arbitrarily large arrays without multiple kernel launches. Accumulates in fast registers before writing to shared memory.

---

## 4. 2D Thread Block Organization
**Example:** `color_inversion/naive.cu`, `count_equal_2d/naive.cu`, `matrix_multiplication/naive.cu`, `matrx_transpose/naive.cu`

Maps 2D thread blocks naturally to 2D data structures:

```cuda
dim3 threadsPerBlock(16, 16);  // 256 threads total
dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

**Benefit:** Clearer code for 2D operations, better memory coalescing for row-adjacent threads.

---

## 5. Shared Memory for Block-Level Aggregation
**Example:** `count_equal/naive.cu`, `count_equal/tree_reduction.cu`, `tensor_reduction_1d/tree_reduction.cu`

Uses fast on-chip shared memory for intermediate results:

```cuda
__shared__ int blockSum;  // ~100x faster than global memory
```

**Benefit:** Shared memory has much lower latency (~20-30 cycles vs ~400-800 cycles for global memory) and higher bandwidth.

---

## 6. Compile-Time Constant Optimization
**Example:** `geglu/naive.cu`

Precomputes constants at compile time to eliminate runtime calculations:

```cuda
constexpr float inv_sqrt2 = 0.7071067811865475f;
// Instead of: float inv_sqrt2 = 1.0f / sqrtf(2.0f);
```

**Benefit:** Saves expensive floating-point operations (division, sqrt) in every thread execution.

---

## 7. Built-in Mathematical Functions
**Example:** `geglu/naive.cu`

Uses CUDA's hardware-accelerated special function units (SFUs):

```cuda
float gelu = 0.5f * x * (1.0f + erff(x * inv_sqrt2));  // erff() uses SFU
```

**Benefit:** Functions like `erff()`, `sinf()`, `cosf()`, `expf()` are hardware-accelerated on GPUs, much faster than software implementations.

---

## 8. Sign Bit Extraction via Bit Manipulation
**Example:** `relu/naive.cu`

Checks float sign using integer bit operations instead of floating-point comparison:

```cuda
if ((__float_as_uint(input[x]) >> 31) != 0) {
    output[x] = 0.0f;  // Negative
    return;
}
```

**Benefit:** Integer operations can be cheaper than FP comparisons on some GPU architectures. The sign bit is bit 31 in IEEE 754 format.

---

## 9. Memory Coalescing
**Example:** `count_equal/tree_reduction.cu`, `tensor_reduction_1d/tree_reduction.cu`

Ensures consecutive threads access consecutive memory addresses:

```cuda
sdata[tid] = (i < N) ? input[i] : 0;  // Threads 0-31 access input[0-31]
```

**Benefit:** GPU memory systems coalesce 32-thread warps into single memory transactions, maximizing bandwidth utilization.

---

## 10. Register-Level Accumulation
**Example:** `tensor_reduction_1d/tree_reduction.cu`

Accumulates intermediate results in registers (fastest memory tier):

```cuda
float local = 0.0f;  // Register variable
for (int i = tid; i < N; i += stride) {
    local += input[i];  // Accumulate in register
}
// Only write to shared memory once
sdata[threadIdx.x] = local;
```

**Benefit:** Registers have single-cycle access latency. Reduces shared memory traffic by factor of (elements per thread).

---

## 11. Thread Synchronization with `__syncthreads()`
**Example:** All multi-threaded block operations

Ensures all threads in a block reach synchronization points before proceeding:

```cuda
if (threadIdx.x == 0) shared_var = 0;
__syncthreads();  // Wait for initialization

// All threads can now safely use shared_var
atomicAdd(&shared_var, value);

__syncthreads();  // Wait for all updates
if (threadIdx.x == 0) finalizeResult();
```

**Benefit:** Critical for correctness in shared memory operations. Prevents race conditions and data hazards.

---

## 12. Single-Thread Global Update
**Example:** All reduction patterns

Minimizes global memory atomic operations by designating one thread per block:

```cuda
if (threadIdx.x == 0) {  // Or (threadIdx.x == 0 && threadIdx.y == 0) for 2D
    atomicAdd(global_output, block_result);
}
```

**Benefit:** Reduces global atomics from (threads_per_block × num_blocks) to just (num_blocks).

---

## Summary Table

| Technique | Key Benefit | Performance Impact |
|-----------|-------------|-------------------|
| Two-Level Atomic Reduction | Reduces global atomic contention | ~10-100x fewer global atomics |
| Tree Reduction | Eliminates intra-block atomics | 2-5x faster than atomic approach |
| Grid-Stride Loop | Handles large arrays efficiently | Enables processing of arbitrarily large data |
| 2D Thread Organization | Natural mapping to 2D data | Better code clarity and memory access |
| Shared Memory | Fast intermediate storage | ~100x faster than global memory |
| Compile-Time Constants | Eliminates runtime computation | Saves expensive operations per thread |
| Built-in Math Functions | Hardware acceleration | 2-10x faster than software implementations |
| Sign Bit Extraction | Faster comparisons | Minor speedup, architecture-dependent |
| Memory Coalescing | Maximizes bandwidth | Up to 32x memory throughput |
| Register Accumulation | Fastest memory tier | Reduces memory traffic significantly |
| Synchronization | Ensures correctness | Required for shared memory operations |
| Single-Thread Updates | Minimizes global operations | O(blocks) instead of O(threads) |
