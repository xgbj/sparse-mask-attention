/*
 * Sparse Mask Attention - CUDA Kernel
 *
 * Round 15: cp.async double-buffering + smem_p bank-conflict fix
 *   (builds on Round 14: WMMA for both QK^T and PV)
 *
 * Architecture:
 *   - 4 warps per block, each warp handles 16 rows of Q (BM=16)
 *   - WMMA m16n16k16: QK^T 4 tiles over HD=64 → float acc
 *   - Softmax: scalar over BN=16 scores per row
 *   - WMMA m16n16k16: PV 4 tiles over HD=64 columns → float acc_frag[4]
 *   - Fragment rescale via smem_rsc[warp][row]
 *   - Output written directly from fragment elements to global memory
 *
 * Round 15 changes:
 *   1. smem_k and smem_v upgraded to double-buffer [2][BN][HD]
 *      cp.async.cg (16B) loads next tile while current tile is computed
 *   2. smem_p padded to [NWARPS][BM][BN+1] to eliminate 2-way bank conflict
 *      during softmax scalar read/write (each row is BN+1 half = odd width)
 *
 * Key WMMA accumulator element layout (m16n16k16, float, row_major):
 *   frag.x[i], lane L:
 *     row = L/4 + ((i & 2) ? 8 : 0)
 *     col offsets: {0,1,0,1,8,9,8,9}[i] + (L%4)*2
 *
 * Shared memory per block:
 *   smem_q[4][16][64]    fp16  =  8192 B
 *   smem_k[2][16][64]    fp16  =  4096 B  (double buffer)
 *   smem_v[2][16][64]    fp16  =  4096 B  (double buffer)
 *   smem_p[4][16][17]    fp16  =  2176 B  (BN+1 padding)
 *   smem_max[4][16]      float =   256 B
 *   smem_sum[4][16]      float =   256 B
 *   smem_rsc[4][16]      float =   256 B
 *   warp_ballots[4]      u32   =    16 B
 *   Total ≈ 19.3 KB
 */

#include "sparse_attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// ---- cp.async helpers (requires sm_80+) ----
// Copy 16 bytes asynchronously from global to shared memory.
// Both src and dst must be 16-byte aligned.
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        : : "r"(smem_addr), "l"(gmem_ptr)
    );
}

// Commit current cp.async group
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" : :);
}

// Wait until at most N async groups remain in flight
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// WMMA accumulator m16n16k16 float element layout (row_major, verified on sm_86):
//   frag.x[i], lane L:
//     row = L/4 + ((i & 2) ? 8 : 0)   [i in {2,3,6,7} → +8]
//     col = (L%4)*2 + (i/4)*8 + (i%2)
//   col_off[i] = {0, 1, 0, 1, 8, 9, 8, 9}
static __device__ __constant__ int wmma_col_off[8] = {0, 1, 0, 1, 8, 9, 8, 9};

__global__ void sparse_attn_wmma_full_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const uint32_t* __restrict__ mask_packed,
    __half* __restrict__ Out,
    float*  __restrict__ LSE,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const bool save_lse
) {
    constexpr int BM     = 16;
    constexpr int BN     = 16;
    constexpr int HD     = 64;
    constexpr int WK     = 16;
    constexpr int NF     = HD / BN;   // = 4 output column fragments per row
    constexpr int NWARPS = 4;
    constexpr int BROWS  = BM * NWARPS;  // = 64
    constexpr int NTH    = 32 * NWARPS;  // = 128

    const int batch_id = blockIdx.z / num_heads;
    const int head_id  = blockIdx.z % num_heads;
    const int blk_m    = blockIdx.x;
    const int blk_row  = blk_m * BROWS;
    if (blk_row >= seq_len) return;

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lane = tid % 32;
    const int wr   = blk_row + wid * BM;   // this warp's first row

    const int bh_off    = (batch_id * num_heads + head_id) * seq_len;
    const __half* Qbh   = Q   + bh_off * HD;
    const __half* Kbh   = K   + bh_off * HD;
    const __half* Vbh   = V   + bh_off * HD;
    __half*       Obh   = Out + bh_off * HD;

    const int nw  = (seq_len + 31) / 32;
    const uint32_t* Mbh = mask_packed + (batch_id * num_heads + head_id) * seq_len * nw;

    // ---- Shared memory (Round 15: double-buffered K/V, padded P) ----
    // PP=8: pad smem_p column dimension so BN+PP = 24 halves per row (48 bytes).
    // This serves two purposes:
    //   1. Bank conflict elimination: 24 halves = 24 banks, odd row stride.
    //   2. WMMA stride alignment: load_matrix_sync requires stride ≡ 0 (mod 8) for __half.
    //      BN+PP = 16+8 = 24, which is divisible by 8. ✓
    constexpr int PP = 8;  // smem_p column padding (24 halves = 48 bytes per row)
    __shared__ __align__(16) __half smem_q[NWARPS][BM][HD];           // Q tiles (16B aligned for WMMA)
    __shared__ __align__(16) __half smem_k[2][BN][HD];               // K double-buffer (16B aligned)
    __shared__ __align__(16) __half smem_v[2][BN][HD];               // V double-buffer (16B aligned)
    __shared__ __align__(16) __half smem_p[NWARPS][BM][BN + PP];     // softmax weights P (16B aligned for WMMA)
    __shared__ float     smem_max[NWARPS][BM];             // running row max
    __shared__ float     smem_sum[NWARPS][BM];             // running row sum
    __shared__ float     smem_rsc[NWARPS][BM];             // rescale factor
    __shared__ uint32_t  warp_ballots[NWARPS];

    // ---- Initialize Q tile and rowstate ----
    for (int i = tid; i < NWARPS * BM * HD; i += NTH) {
        int w = i / (BM * HD);
        int r = (i % (BM * HD)) / HD;
        int d = i % HD;
        int gr = blk_row + w * BM + r;
        smem_q[w][r][d] = (gr < seq_len) ? Qbh[gr * HD + d] : __float2half(0.f);
    }
    if (lane < BM) {
        smem_max[wid][lane] = -65504.f;
        smem_sum[wid][lane] = 0.f;
        smem_rsc[wid][lane] = 1.f;
    }

    // ---- Per-warp accumulator fragments ----
    wmma::fragment<wmma::accumulator, BM, BN, WK, float> acc_frag[NF];
#pragma unroll
    for (int f = 0; f < NF; f++) wmma::fill_fragment(acc_frag[f], 0.f);

    const int lrow = lane % BM;
    const int grow = wr + lrow;
    const bool vr  = (lane < BM) && (wr < seq_len) && (grow < seq_len);

    const int ntiles = (seq_len + BN - 1) / BN;

    // ---- cp.async pipeline: each thread handles 1 chunk of 8 halves (16B) per K and V ----
    // BN*HD = 16*64 = 1024 halves = 128 chunks. NTH=128, so 1 chunk per thread per buffer.
    // chunk c = tid: row = tid/(HD/8) = tid/8, col_base = (tid%(HD/8))*8 = (tid%8)*8
    // Addresses: Kbh[gc*HD + d] where gc=cs+row, d=col_base (8-half aligned → 16B aligned).
    // smem_k[buf][row][d]: smem_k is .align 16, each row HD*2=128B, d multiples of 8 → 16B aligned.
#define LOAD_KV_ASYNC(tile_n_, buf_) do { \
    { \
        const int _cs = (tile_n_) * BN; \
        const int _r  = tid / (HD / 8); \
        const int _d  = (tid % (HD / 8)) * 8; \
        const int _gc = _cs + _r; \
        if (_gc < seq_len) { \
            cp_async_16B(&smem_k[(buf_)][_r][_d], &Kbh[_gc * HD + _d]); \
            cp_async_16B(&smem_v[(buf_)][_r][_d], &Vbh[_gc * HD + _d]); \
        } else { \
            smem_k[(buf_)][_r][_d+0] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+1] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+2] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+3] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+4] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+5] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+6] = __float2half(0.f); \
            smem_k[(buf_)][_r][_d+7] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+0] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+1] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+2] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+3] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+4] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+5] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+6] = __float2half(0.f); \
            smem_v[(buf_)][_r][_d+7] = __float2half(0.f); \
        } \
    } \
} while(0)

    // ---- Single-buffer cp.async: async load at start of each iteration ----
    // cp.async.cg uses L2 cache bypass path, potentially faster than regular loads.
    // No double-buffering pipeline complexity; simple load-then-wait pattern.
    // cbuf is always 0 (single buffer), but keeping as (tn&1) for potential future upgrade.
    __syncthreads();  // Q load complete

    // ---- Main loop ----
    for (int tn = 0; tn < ntiles; tn++) {
        const int cs   = tn * BN;
        const int cbuf = 0;  // single buffer (no double-buffering for simplicity)

        // Load K and V for current tile asynchronously
        LOAD_KV_ASYNC(tn, cbuf);
        cp_async_commit();

        // ---- Sparse block skip check ----
        uint32_t mword = 0u;
        if (vr) {
            int wi = cs / 32, bo = cs % 32;
            mword = Mbh[grow * nw + wi] >> bo;
            if (bo + BN > 32 && (wi + 1) < nw)
                mword |= Mbh[grow * nw + wi + 1] << (32 - bo);
            int vc = min(BN, seq_len - cs);
            if (vc < 32) mword &= (1u << vc) - 1u;
        }
        uint32_t ballot = __ballot_sync(0xffffffff, mword != 0u);
        if (lane == 0) warp_ballots[wid] = ballot;

        // Wait for current tile's load to complete before using it
        cp_async_wait<0>();
        __syncthreads();

        if ((warp_ballots[0] | warp_ballots[1] | warp_ballots[2] | warp_ballots[3]) == 0u) continue;

        // ---- WMMA QK^T → write scores to smem_p (fp16, with mask) ----
        if (wr < seq_len) {
            wmma::fragment<wmma::accumulator, BM, BN, WK, float> qk;
            wmma::fill_fragment(qk, 0.f);
#pragma unroll
            for (int kk = 0; kk < HD; kk += WK) {
                wmma::fragment<wmma::matrix_a, BM, BN, WK, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, BM, BN, WK, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, &smem_q[wid][0][kk], HD);
                wmma::load_matrix_sync(b, &smem_k[cbuf][0][kk], HD);
                wmma::mma_sync(qk, a, b, qk);
            }

            // Write scores to smem_p with mask applied
            // Fragment layout: frag.x[i] at (row=lane/4+((i&2)?8:0), col=(lane%4)*2+col_off[i])
#pragma unroll
            for (int i = 0; i < 8; i++) {
                int fr  = lane / 4 + ((i & 2) ? 8 : 0);
                int fc  = (lane % 4) * 2 + wmma_col_off[i];
                int gr2 = wr + fr;
                int gc2 = cs + fc;
                bool ok = (gr2 < seq_len) && (gc2 < seq_len) &&
                          ((Mbh[gr2 * nw + gc2 / 32] >> (gc2 % 32)) & 1u);
                float sv = ok ? (qk.x[i] * scale) : -65504.f;
                smem_p[wid][fr][fc] = __float2half(sv);
            }
        }
        __syncwarp();   // smem_p written by all 32 lanes in this warp

        // ---- Softmax (lanes 0..BM-1 each handle one row) ----
        if (vr) {
            float old_max = smem_max[wid][lrow];
            float old_sum = smem_sum[wid][lrow];

            float sc[BN];
#pragma unroll
            for (int j = 0; j < BN; j++) sc[j] = __half2float(smem_p[wid][lrow][j]);

            float nm = old_max;
#pragma unroll
            for (int j = 0; j < BN; j++) nm = fmaxf(nm, sc[j]);

            float rsc = (old_max > -65000.f) ? expf(old_max - nm) : 1.f;
            smem_rsc[wid][lrow] = rsc;

            float ns = old_sum * rsc;

            float pv[BN];
#pragma unroll
            for (int j = 0; j < BN; j++) {
                pv[j] = (sc[j] > -65000.f) ? expf(sc[j] - nm) : 0.f;
                ns += pv[j];
            }

            smem_max[wid][lrow] = nm;
            smem_sum[wid][lrow] = ns;

            // Write P (fp16) for WMMA PV — stride = BN+PP
#pragma unroll
            for (int j = 0; j < BN; j++) smem_p[wid][lrow][j] = __float2half(pv[j]);
        }
        __syncwarp();   // smem_rsc and smem_p now valid

        // ---- Rescale acc_frag using smem_rsc ----
        if (wr < seq_len) {
#pragma unroll
            for (int f = 0; f < NF; f++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    int fr = lane / 4 + ((i & 2) ? 8 : 0);
                    acc_frag[f].x[i] *= smem_rsc[wid][fr];
                }
            }
        }

        // ---- WMMA PV: P[BM×BN] × V[BN×HD] → acc_frag[NF] ----
        // smem_p stride = BN+PP (padded row width)
        if (wr < seq_len) {
#pragma unroll
            for (int f = 0; f < NF; f++) {
                wmma::fragment<wmma::matrix_a, BM, BN, WK, __half, wmma::row_major> pa;
                wmma::fragment<wmma::matrix_b, BM, BN, WK, __half, wmma::row_major> vb;
                wmma::load_matrix_sync(pa, &smem_p[wid][0][0], BN + PP);
                wmma::load_matrix_sync(vb, &smem_v[cbuf][0][f * BN], HD);
                wmma::mma_sync(acc_frag[f], pa, vb, acc_frag[f]);
            }
        }

        // cp_async pipeline: next tile was already issued at the top of this iteration.
        // No explicit sync needed here; the next iteration's cp_async_wait<0> handles it.
    }

    // ---- Normalize and write output directly from fragment elements ----
    if (wr < seq_len) {
#pragma unroll
        for (int f = 0; f < NF; f++) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                int fr   = lane / 4 + ((i & 2) ? 8 : 0);
                int fc   = (lane % 4) * 2 + wmma_col_off[i];
                int gr2  = wr + fr;
                int gc2  = f * BN + fc;
                if (gr2 < seq_len) {
                    float s = smem_sum[wid][fr];
                    float out_val = (s > 0.f) ? (acc_frag[f].x[i] / s) : 0.f;
                    Obh[gr2 * HD + gc2] = __float2half(out_val);
                }
                if (save_lse && LSE && f == 0 && i == 0) {
                    // fc==0 when lane%4==0 && wmma_col_off[0]==0, but i==0 always fc=col_off[0]=0
                    // Only write LSE for fc=0 column (one thread per row)
                    if (fc == 0 && gr2 < seq_len) {
                        int li = (batch_id * num_heads + head_id) * seq_len + gr2;
                        float m = smem_max[wid][fr];
                        float s = smem_sum[wid][fr];
                        LSE[li] = m + logf(fmaxf(s, 1e-30f));
                    }
                }
            }
        }
    }
}

// ============================================================
// Scalar kernel (BF16 fallback)
// ============================================================

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int NUM_THREADS, typename scalar_t>
__global__ void sparse_attention_fwd_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const uint32_t* __restrict__ mask_packed,
    scalar_t* __restrict__ Out,
    float*   __restrict__ LSE,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const bool save_lse
) {
    const int batch_id = blockIdx.z / num_heads;
    const int head_id  = blockIdx.z % num_heads;
    const int tile_m   = blockIdx.x;
    const int row_start = tile_m * BLOCK_M;
    if (row_start >= seq_len) return;

    __shared__ float smem_q[BLOCK_M][HEAD_DIM];
    __shared__ float smem_k[BLOCK_N][HEAD_DIM];
    __shared__ float smem_v[BLOCK_N][HEAD_DIM];

    const int tid = threadIdx.x;
    const int bh_offset = (batch_id * num_heads + head_id) * seq_len;
    const scalar_t* Q_bh = Q + bh_offset * HEAD_DIM;
    const scalar_t* K_bh = K + bh_offset * HEAD_DIM;
    const scalar_t* V_bh = V + bh_offset * HEAD_DIM;
    scalar_t*       O_bh = Out + bh_offset * HEAD_DIM;
    const int n_words = (seq_len + 31) / 32;
    const uint32_t* mask_bh = mask_packed + (batch_id * num_heads + head_id) * seq_len * n_words;

    {
        constexpr int TF = BLOCK_M * HEAD_DIM / 4;
        for (int i = tid; i < TF; i += NUM_THREADS) {
            int fl = i * 4, r = fl / HEAD_DIM, d = fl % HEAD_DIM, gr = row_start + r;
            if (gr < seq_len) {
                const scalar_t* s = Q_bh + gr * HEAD_DIM + d;
                smem_q[r][d]=to_float(s[0]); smem_q[r][d+1]=to_float(s[1]);
                smem_q[r][d+2]=to_float(s[2]); smem_q[r][d+3]=to_float(s[3]);
            }
        }
    }
    __syncthreads();

    const int local_row = tid, global_row = row_start + local_row;
    const bool is_compute = (tid < BLOCK_M), valid_row = is_compute && (global_row < seq_len);
    float row_max = -FLT_MAX, row_sum = 0.0f;
    float acc[HEAD_DIM];
    if (is_compute) { for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f; }

    for (int tile_n = 0; tile_n < (seq_len + BLOCK_N - 1) / BLOCK_N; tile_n++) {
        const int col_start = tile_n * BLOCK_N, col_end = min(col_start + BLOCK_N, seq_len);
        uint32_t mask_word = 0u;
        if (valid_row) {
            int wi = col_start / 32, bo = col_start % 32;
            uint32_t raw = mask_bh[global_row * n_words + wi];
            mask_word = raw >> bo;
            if (bo + BLOCK_N > 32 && (wi + 1) < n_words)
                mask_word |= mask_bh[global_row * n_words + wi + 1] << (32 - bo);
            int vc = min(BLOCK_N, seq_len - col_start);
            if (vc < 32) mask_word &= (1u << vc) - 1u;
        }
        __shared__ uint32_t sb;
        if (tid < 32) { uint32_t b = __ballot_sync(0xffffffff, mask_word != 0u); if (tid == 0) sb = b; }
        __syncthreads();
        if (sb == 0u) continue;

        {
            int nc = col_end - col_start, tf = nc * HEAD_DIM / 4;
            for (int i = tid; i < tf; i += NUM_THREADS) {
                int fl = i * 4, r = fl / HEAD_DIM, d = fl % HEAD_DIM, gc = col_start + r;
                const scalar_t* ks = K_bh + gc * HEAD_DIM + d;
                const scalar_t* vs = V_bh + gc * HEAD_DIM + d;
                smem_k[r][d]=to_float(ks[0]); smem_k[r][d+1]=to_float(ks[1]);
                smem_k[r][d+2]=to_float(ks[2]); smem_k[r][d+3]=to_float(ks[3]);
                smem_v[r][d]=to_float(vs[0]); smem_v[r][d+1]=to_float(vs[1]);
                smem_v[r][d+2]=to_float(vs[2]); smem_v[r][d+3]=to_float(vs[3]);
            }
        }
        __syncthreads();

        if (is_compute) {
            float scores[BLOCK_N];
#pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                if (col_start + j >= seq_len || !valid_row || !((mask_word >> j) & 1u)) {
                    scores[j] = -FLT_MAX; continue;
                }
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    dot += smem_q[local_row][d]*smem_k[j][d] + smem_q[local_row][d+1]*smem_k[j][d+1]
                         + smem_q[local_row][d+2]*smem_k[j][d+2] + smem_q[local_row][d+3]*smem_k[j][d+3];
                }
                scores[j] = dot * scale;
            }
            float nm = row_max;
#pragma unroll
            for (int j = 0; j < BLOCK_N; j++) nm = fmaxf(nm, scores[j]);
            float rs = expf(row_max - nm); row_sum *= rs;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) acc[d] *= rs;
            row_max = nm;
#pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                if (scores[j] == -FLT_MAX) continue;
                float p = expf(scores[j] - row_max); row_sum += p;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) acc[d] += p * smem_v[j][d];
            }
        }
        __syncthreads();
    }
    if (valid_row) {
        float is = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        scalar_t* op = O_bh + global_row * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) op[d] = from_float<scalar_t>(acc[d] * is);
        if (save_lse && LSE) {
            int li = (batch_id * num_heads + head_id) * seq_len + global_row;
            LSE[li] = row_max + logf(row_sum);
        }
    }
}

// ============================================================
// Mask packing
// ============================================================

__global__ void pack_mask_kernel(
    const bool* __restrict__ mask,
    uint32_t*   __restrict__ mask_packed,
    int B, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_words = (N + 31) / 32;
    int total = B * H * N * n_words;
    if (idx >= total) return;
    int word = idx % n_words, tmp = idx / n_words;
    int row = tmp % N; tmp /= N;
    int h = tmp % H, b = tmp / H;
    int cs = word * 32;
    uint32_t packed = 0u;
    for (int bit = 0; bit < 32 && (cs + bit) < N; bit++) {
        bool val = mask[((b * H + h) * N + row) * N + cs + bit];
        packed |= ((uint32_t)val << bit);
    }
    mask_packed[idx] = packed;
}

// ============================================================
// Host launchers
// ============================================================

void sparse_attention_fwd_bf16(
    const SparseAttentionParams& params,
    cudaStream_t stream
) {
    const int BLOCK_M = 32, BLOCK_N = 16, HEAD_DIM = 64, NUM_THREADS = 128;
    int ntm = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(ntm, 1, params.batch_size * params.num_heads);
    sparse_attention_fwd_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, NUM_THREADS, __nv_bfloat16>
        <<<grid, NUM_THREADS, 0, stream>>>(
            (const __nv_bfloat16*)params.q, (const __nv_bfloat16*)params.k,
            (const __nv_bfloat16*)params.v, params.mask_packed,
            (__nv_bfloat16*)params.out, params.lse, params.scale,
            params.batch_size, params.num_heads, params.seq_len, params.is_training);
}

void sparse_attention_fwd_fp16(
    const SparseAttentionParams& params,
    cudaStream_t stream
) {
    const int BROWS = 64;  // 4 warps × 16 rows
    int ntm = (params.seq_len + BROWS - 1) / BROWS;
    dim3 grid(ntm, 1, params.batch_size * params.num_heads);
    sparse_attn_wmma_full_fp16<<<grid, 128, 0, stream>>>(
        (const __half*)params.q, (const __half*)params.k,
        (const __half*)params.v, params.mask_packed,
        (__half*)params.out, params.lse, params.scale,
        params.batch_size, params.num_heads, params.seq_len, params.is_training);
}

void pack_mask_bits(
    const bool* mask, uint32_t* mask_packed,
    int B, int H, int N, cudaStream_t stream
) {
    int n_words = (N + 31) / 32;
    int total = B * H * N * n_words;
    pack_mask_kernel<<<(total + 255) / 256, 256, 0, stream>>>(mask, mask_packed, B, H, N);
}
