#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>

// ============================================================
// 基础工具宏
// ============================================================

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ============================================================
// Warp 级别 reduce 工具
// ============================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================
// Bit-packed mask 读取工具
// ============================================================

// 读取 mask[b][h][row][col]，mask 已按 uint32 打包
__device__ __forceinline__ bool read_mask_bit(
    const uint32_t* mask_packed,
    int b, int h, int row, int col,
    int num_heads, int seq_len
) {
    int n_words = CEIL_DIV(seq_len, 32);
    int idx = ((b * num_heads + h) * seq_len + row) * n_words + (col / 32);
    uint32_t word = mask_packed[idx];
    return (word >> (col % 32)) & 1u;
}

// 读取一整个 uint32 word，包含 32 个连续 mask 位
__device__ __forceinline__ uint32_t read_mask_word(
    const uint32_t* mask_packed,
    int b, int h, int row, int col_word,
    int num_heads, int seq_len
) {
    int n_words = CEIL_DIV(seq_len, 32);
    int idx = ((b * num_heads + h) * seq_len + row) * n_words + col_word;
    return mask_packed[idx];
}

// 判断一个 block [row_start, row_end) x [col_start, col_end) 是否全为 0
// 用于 sparse block skipping
__device__ __forceinline__ bool is_block_all_masked(
    const uint32_t* mask_packed,
    int b, int h,
    int row_start, int row_end,
    int col_start, int col_end,
    int num_heads, int seq_len
) {
    int n_words = CEIL_DIV(seq_len, 32);
    int word_start = col_start / 32;
    int word_end   = CEIL_DIV(col_end, 32);

    for (int row = row_start; row < row_end; row++) {
        int base = ((b * num_heads + h) * seq_len + row) * n_words;
        for (int w = word_start; w < word_end; w++) {
            if (mask_packed[base + w] != 0u) return false;
        }
    }
    return true;
}

// ============================================================
// 数值类型转换工具
// ============================================================

__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ float to_float(__half x) {
    return __half2float(x);
}
__device__ __forceinline__ float to_float(float x) {
    return x;
}

__device__ __forceinline__ __nv_bfloat16 from_float_bf16(float x) {
    return __float2bfloat16(x);
}
__device__ __forceinline__ __half from_float_fp16(float x) {
    return __float2half(x);
}

// ============================================================
// Shared memory 布局辅助
// ============================================================

// 对 shared memory bank conflict 做 padding
// 每行加 1 个 float 的 padding，避免 32-bank conflict
template<int COLS>
struct PaddedShared {
    static constexpr int PADDED_COLS = COLS + 1;
};
