# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
pip install -r requirements.txt
pip install -e .
```

Compiles `csrc/sparse_attention.cu` into the `sparse_attn_cuda` Python extension. Targets A100 (`sm_80`) with `-O3 --use_fast_math`.

## Tests

```bash
python tests/test_correctness.py   # correctness vs PyTorch reference
python tests/test_performance.py   # latency/MFU scaling
```

Benchmarks (require `flash-attn`):
```bash
python benchmarks/profile_mfu.py
python benchmarks/compare_flash.py --seq_len 512
python benchmarks/compare_flashinfer.py --seq_len 512
```

## Architecture

CUDA kernel for sparse masked attention optimized for short sequences (<1K tokens) with high sparsity (~75%) on A100.

**Key optimization:** Bit-packs the bool mask `[B,H,N,N]` → uint32 `[B,H,N,N/32]` (8x memory savings), then skips fully-masked K/V blocks in the kernel using `__ballot_sync()` warp voting.

**Kernel parameters:** `BLOCK_M=32, BLOCK_N=32`, head dim fixed at 64, FP32 accumulation for online softmax, BF16/FP16 I/O.

### File roles

- `csrc/sparse_attention.cu` — forward kernel (`sparse_attention_fwd_kernel`), mask packing kernel, BF16/FP16 host launchers
- `csrc/sparse_attention.h` — `SparseAttentionParams` struct, function declarations
- `csrc/utils.cuh` — warp reductions, bit-mask read helpers (`read_mask_bit`, `is_block_all_masked`), type conversions
- `python/sparse_attention.py` — PyTorch autograd `Function` wrapper, `sparse_attention()` user API, `sparse_attention_ref()` reference impl, `pack_mask()`
- `python/benchmark.py` — MFU measurement utilities, A100 peak TFLOPS lookup

### Data flow

```
Q, K, V [B,H,N,D] + bool mask [B,H,N,N]
  → pack_mask_kernel → uint32 mask [B,H,N,N/32]
  → sparse_attention_fwd_kernel (skips all-masked blocks)
  → output [B,H,N,D]
```

Backward saves LSE (log-sum-exp) from forward for gradient computation.
