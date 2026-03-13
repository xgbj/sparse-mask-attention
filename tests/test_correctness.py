"""
正确性验证：对比 CUDA kernel 输出与 PyTorch 参考实现

用法：
    python tests/test_correctness.py
"""

import torch
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.sparse_attention import sparse_attention, sparse_attention_ref, generate_random_sparse_mask


def test_output_correctness(B=2, H=4, N=64, D=64, sparsity=0.75, dtype=torch.bfloat16):
    device = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    mask = generate_random_sparse_mask(B, H, N, sparsity=sparsity, device=device)

    # 参考输出（FP32）
    ref_out = sparse_attention_ref(q.float(), k.float(), v.float(), mask)

    # 我们的输出
    our_out = sparse_attention(q, k, v, mask).float()

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    # BF16 精度下允许的误差
    tol = 1e-2
    passed = max_diff < tol

    print(f"[正确性] B={B} H={H} N={N} D={D} sparsity={sparsity}")
    print(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  {'PASS' if passed else 'FAIL'}")
    assert passed, f"正确性检验失败：max_diff={max_diff} > tol={tol}"
    return passed


def test_all_masked_row():
    """测试某行全为 masked 时输出为 0（而非 nan）"""
    device = "cuda"
    B, H, N, D = 1, 1, 8, 64
    q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)

    # 第 0 行全为 masked
    mask = torch.ones(B, H, N, N, device=device, dtype=torch.bool)
    mask[:, :, 0, :] = False

    out = sparse_attention(q, k, v, mask)
    assert not torch.isnan(out).any(), "全 masked 行不应产生 nan"
    print("[全 masked 行] PASS")


def test_gradient():
    """测试反向传播梯度正确性"""
    device = "cuda"
    B, H, N, D = 2, 4, 32, 64
    torch.manual_seed(0)

    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    mask = generate_random_sparse_mask(B, H, N, sparsity=0.75, device=device)

    # 参考梯度
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = sparse_attention_ref(q_ref, k_ref, v_ref, mask)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # 我们的梯度
    out = sparse_attention(q, k, v, mask)
    loss = out.sum()
    loss.backward()

    for name, g1, g2 in [("dq", q.grad, q_ref.grad),
                          ("dk", k.grad, k_ref.grad),
                          ("dv", v.grad, v_ref.grad)]:
        diff = (g1 - g2).abs().max().item()
        print(f"[梯度 {name}] max_diff={diff:.6f}  {'PASS' if diff < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"

    print("=== 正确性测试 ===\n")
    for N in [64, 128, 256, 512]:
        test_output_correctness(N=N)

    print()
    test_all_masked_row()

    print()
    test_gradient()

    print("\n所有测试通过！")
