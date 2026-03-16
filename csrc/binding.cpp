#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "sparse_attention.h"

// ============================================================
// PyTorch Tensor -> CUDA kernel 的桥接
// ============================================================

void forward(
    torch::Tensor q,          // [B, H, N, D]
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask_packed, // [B, H, N, N/32] uint32
    torch::Tensor out,
    torch::Tensor lse,         // [B, H, N] float32, 可为空
    float scale,
    bool is_training
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(q.dim() == 4, "q must be 4D");

    int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);

    SparseAttentionParams params;
    params.q = q.data_ptr();
    params.k = k.data_ptr();
    params.v = v.data_ptr();
    params.mask_packed = (const uint32_t*)mask_packed.data_ptr();
    params.out = out.data_ptr();
    params.lse = is_training ? lse.data_ptr<float>() : nullptr;
    params.batch_size = B;
    params.num_heads = H;
    params.seq_len = N;
    params.head_dim = D;
    params.scale = scale;
    params.is_training = is_training;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (q.scalar_type() == at::ScalarType::BFloat16) {
        sparse_attention_fwd_bf16(params, stream);
    } else if (q.scalar_type() == at::ScalarType::Half) {
        sparse_attention_fwd_fp16(params, stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype, need BF16 or FP16");
    }
}

torch::Tensor pack_mask(torch::Tensor mask) {
    TORCH_CHECK(mask.is_cuda(), "mask must be CUDA tensor");
    TORCH_CHECK(mask.dim() == 4, "mask must be 4D [B, H, N, N]");

    int B = mask.size(0), H = mask.size(1), N = mask.size(2);
    int n_words = (N + 31) / 32;

    auto packed = torch::zeros({B, H, N, n_words},
        torch::TensorOptions().dtype(torch::kInt32).device(mask.device()));

    pack_mask_bits(
        (const bool*)mask.data_ptr(),
        (uint32_t*)packed.data_ptr(),
        B, H, N,
        at::cuda::getCurrentCUDAStream()
    );
    return packed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sparse attention forward");
    m.def("pack_mask", &pack_mask, "Pack bool mask to uint32");
}
