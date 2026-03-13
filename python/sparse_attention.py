"""
Sparse Mask Attention - Python 接口

封装 CUDA kernel，提供 PyTorch autograd 支持。
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import math

try:
    import sparse_attn_cuda  # 编译后的 CUDA 扩展
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("[Warning] CUDA 扩展未编译，将使用 PyTorch 参考实现")


# ============================================================
# PyTorch 参考实现（用于正确性验证）
# ============================================================

def sparse_attention_ref(q, k, v, mask):
    """
    参考实现：标准 PyTorch，用于验证 CUDA kernel 正确性。
    mask: bool tensor [B, H, N, N]，True 表示保留，False 表示 mask 掉
    """
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    # 将 mask=False 的位置设为 -inf
    scores = scores.masked_fill(~mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    # 处理全 masked 行（softmax 后为 nan）
    attn = torch.nan_to_num(attn, nan=0.0)

    out = torch.matmul(attn, v)
    return out


# ============================================================
# Autograd Function
# ============================================================

class SparseAttentionFunction(Function):

    @staticmethod
    def forward(ctx, q, k, v, mask_packed, mask_bool, scale, seq_len):
        """
        q, k, v: [B, H, N, D]，BF16 或 FP16
        mask_packed: [B, H, N, N/32]，uint32，bit-packed
        mask_bool: [B, H, N, N]，bool，用于反向传播
        """
        B, H, N, D = q.shape

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        if HAS_CUDA_EXT:
            sparse_attn_cuda.forward(q, k, v, mask_packed, out, lse, scale, True)
        else:
            # fallback 到参考实现
            out_ref = sparse_attention_ref(
                q.float(), k.float(), v.float(), mask_bool
            ).to(q.dtype)
            out.copy_(out_ref)
            # 计算 lse（用于反向传播）
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~mask_bool, float('-inf'))
            lse.copy_(torch.logsumexp(scores, dim=-1))

        ctx.save_for_backward(q, k, v, out, lse, mask_packed, mask_bool)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, mask_packed, mask_bool = ctx.saved_tensors
        scale = ctx.scale

        B, H, N, D = q.shape
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        if HAS_CUDA_EXT:
            sparse_attn_cuda.backward(
                dout, q, k, v, out, lse, mask_packed,
                dq, dk, dv, scale
            )
        else:
            # 参考反向传播
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~mask_bool, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            dv.copy_(torch.matmul(attn.transpose(-2, -1), dout.float()).to(dv.dtype))
            dattn = torch.matmul(dout.float(), v.float().transpose(-2, -1))
            dattn = dattn.masked_fill(~mask_bool, 0.0)

            # softmax 反向
            ds = attn * (dattn - (dattn * attn).sum(dim=-1, keepdim=True))
            ds = ds * scale

            dq.copy_(torch.matmul(ds, k.float()).to(dq.dtype))
            dk.copy_(torch.matmul(ds.transpose(-2, -1), q.float()).to(dk.dtype))

        return dq, dk, dv, None, None, None, None


# ============================================================
# 主接口函数
# ============================================================

def sparse_attention(q, k, v, mask):
    """
    高性能稀疏 mask 注意力。

    参数：
        q, k, v: Tensor [B, H, N, D]，支持 BF16/FP16/FP32
        mask: bool Tensor [B, H, N, N]，True=保留，False=mask 掉
              期望稀疏度约 75%（即 75% 为 False）

    返回：
        out: Tensor [B, H, N, D]，与输入相同 dtype
    """
    assert q.shape == k.shape == v.shape, "q/k/v 形状必须一致"
    assert q.device == k.device == v.device == mask.device, "所有张量必须在同一设备"
    assert mask.dtype == torch.bool, "mask 必须是 bool 类型"

    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # 转换为 BF16（如果是 FP32 输入）
    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    if HAS_CUDA_EXT and q.is_cuda:
        # 压缩 mask 为 bit-packed 格式
        mask_packed = pack_mask(mask)
        out = SparseAttentionFunction.apply(q, k, v, mask_packed, mask, scale, N)
    else:
        out = sparse_attention_ref(q.float(), k.float(), v.float(), mask).to(q.dtype)

    if orig_dtype == torch.float32:
        out = out.float()

    return out


def sparse_attention_func(q, k, v, mask):
    """sparse_attention 的别名，不带 autograd（推理专用，略快）"""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    with torch.no_grad():
        if HAS_CUDA_EXT and q.is_cuda:
            mask_packed = pack_mask(mask)
            out = torch.empty_like(q)
            lse = None
            sparse_attn_cuda.forward(q, k, v, mask_packed, out, lse, scale, False)
        else:
            out = sparse_attention_ref(q.float(), k.float(), v.float(), mask).to(q.dtype)

    if orig_dtype == torch.float32:
        out = out.float()
    return out


# ============================================================
# Mask 工具函数
# ============================================================

def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    将 bool mask [B, H, N, N] 压缩为 bit-packed uint32 [B, H, N, N/32]
    节省 8x 内存带宽。
    """
    B, H, N, _ = mask.shape
    if HAS_CUDA_EXT:
        return sparse_attn_cuda.pack_mask(mask)
    else:
        # CPU fallback
        n_words = (N + 31) // 32
        packed = torch.zeros(B, H, N, n_words, dtype=torch.int32, device=mask.device)
        for w in range(n_words):
            col_start = w * 32
            col_end = min(col_start + 32, N)
            for bit, col in enumerate(range(col_start, col_end)):
                packed[:, :, :, w] |= (mask[:, :, :, col].int() << bit)
        return packed


def generate_random_sparse_mask(B, H, N, sparsity=0.75, device='cuda'):
    """
    生成随机稀疏 mask。
    sparsity=0.75 表示 75% 的位置被 mask 掉（False）。
    """
    # rand > sparsity 的概率 = 1 - sparsity = 25% 为 True
    mask = torch.rand(B, H, N, N, device=device) > sparsity
    return mask
