#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ============================================================
// 核心数据结构
// ============================================================

struct SparseAttentionParams {
    // 输入指针
    const void* q;       // [B, H, N, d]
    const void* k;       // [B, H, N, d]
    const void* v;       // [B, H, N, d]
    const uint32_t* mask_packed;  // bit-packed mask [B, H, N, N/32]

    // 输出指针
    void* out;           // [B, H, N, d]
    float* lse;          // log-sum-exp [B, H, N]，训练时需要

    // 形状参数
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;

    // 步长（以元素为单位）
    int stride_qb, stride_qh, stride_qn;
    int stride_kb, stride_kh, stride_kn;
    int stride_vb, stride_vh, stride_vn;
    int stride_ob, stride_oh, stride_on;

    // 超参数
    float scale;         // 1 / sqrt(head_dim)
    bool is_training;    // 是否需要保存 lse 用于反向传播
};

// ============================================================
// Kernel 启动函数声明
// ============================================================

// 前向传播：BF16
void sparse_attention_fwd_bf16(
    const SparseAttentionParams& params,
    cudaStream_t stream
);

// 前向传播：FP16
void sparse_attention_fwd_fp16(
    const SparseAttentionParams& params,
    cudaStream_t stream
);

// 反向传播：BF16
void sparse_attention_bwd_bf16(
    const SparseAttentionParams& params,
    const void* dout,    // [B, H, N, d]
    void* dq,            // [B, H, N, d]
    void* dk,            // [B, H, N, d]
    void* dv,            // [B, H, N, d]
    cudaStream_t stream
);

// Mask 压缩工具
void pack_mask_bits(
    const bool* mask,    // [B, H, N, N]
    uint32_t* mask_packed, // [B, H, N, N/32]
    int B, int H, int N,
    cudaStream_t stream
);
