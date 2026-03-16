/*
 * Sparse Mask Attention - CUDA Kernel
 *
 * Round 23: Register-resident softmax — eliminate smem_max/sum/rsc,
 *   do softmax entirely in registers using WMMA fragment layout.
 *   (builds on Round 22: merged sc/pv arrays)
 *
 * Key change: Exploit WMMA accumulator layout (row = lane/4, 4 lanes/row)
 *   to do softmax via __shfl_xor_sync(mask, val, 1/2) cross-lane reduction.
 *   - Eliminates QK→smem_p write + softmax smem_p read (50% less smem_p traffic)
 *   - Eliminates smem_max/smem_sum/smem_rsc (all register-resident)
 *   - Removes 1 __syncwarp barrier per iteration
 *   - Per-iteration savings: ~96 smem load/store + 1 barrier
 *
 * WMMA accumulator element layout (m16n16k16, float, row_major):
 *   frag.x[i], lane L:
 *     row = L/4 + ((i & 2) ? 8 : 0)
 *     col offsets: {0,1,0,1,8,9,8,9}[i] + (L%4)*2
 *
 * Shared memory per block:
 *   smem_q[4][16][72]    fp16  =  9216 B  (HD+8 padding)
 *   smem_k[64][72]       fp16  =  9216 B  (HD+8 padding)
 *   smem_v[64][72]       fp16  =  9216 B  (HD+8 padding)
 *   smem_p[4][16][72]    fp16  =  9216 B  (BN+8 padding)
 *   smem_mask[64][2]     u32   =   512 B
 *   warp_ballots[4]      u32   =    16 B
 *   Total ≈ 36.7 KB
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
    constexpr int BN     = 64;   // larger tile: 4x fewer loop iterations
    constexpr int HD     = 64;
    constexpr int WK     = 16;
    constexpr int NQK    = BN / WK;  // = 4 WMMA tiles for QK^T (BM×WK × WK×BN)
    constexpr int NF     = HD / WK;  // = 4 output column fragments per row (PV)
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

    // ---- Shared memory (Round 19: pad K/V inner dim to eliminate bank conflicts) ----
    // PP=8: pad smem_p/q/k/v columns so stride = 72 halves (144 bytes).
    //   144 / 4 = 36 bank slots per row, 36 % 32 = 4 → consecutive rows shift 4 banks.
    //   Reduces 16-way bank conflicts to 2-way for WMMA loads.
    constexpr int PP = 8;  // column padding for all smem arrays
    __shared__ __align__(16) __half smem_q[NWARPS][BM][HD + PP];     // Q tiles (padded)
    __shared__ __align__(16) __half smem_k[BN][HD + PP];             // K single-buffer (padded)
    __shared__ __align__(16) __half smem_v[BN][HD + PP];             // V single-buffer (padded)
    __shared__ __align__(16) __half smem_p[NWARPS][BM][BN + PP];    // scores P (padded)
    __shared__ uint32_t  warp_ballots[NWARPS];
    __shared__ uint32_t  smem_mask[BROWS][2];  // mask bits for current tile: [row][word0/word1]

    // ---- Initialize Q tile and register-resident row state ----
    for (int i = tid; i < NWARPS * BM * HD; i += NTH) {
        int w = i / (BM * HD);
        int r = (i % (BM * HD)) / HD;
        int d = i % HD;
        int gr = blk_row + w * BM + r;
        smem_q[w][r][d] = (gr < seq_len) ? Qbh[gr * HD + d] : __float2half(0.f);
    }

    // Running max/sum in registers: each lane tracks 2 rows (row_a=lane/4, row_b=lane/4+8)
    float old_max_a = -65504.f, old_max_b = -65504.f;
    float old_sum_a = 0.f, old_sum_b = 0.f;

    // ---- Per-warp accumulator fragments ----
    // NF=4 fragments, each m16n16k16, covering output cols [f*16..(f+1)*16-1]
    wmma::fragment<wmma::accumulator, BM, WK, WK, float> acc_frag[NF];
#pragma unroll
    for (int f = 0; f < NF; f++) wmma::fill_fragment(acc_frag[f], 0.f);

    const int lrow = lane % BM;
    const int grow = wr + lrow;
    const bool vr  = (lane < BM) && (wr < seq_len) && (grow < seq_len);

    const int ntiles = (seq_len + BN - 1) / BN;

    // ---- K/V load: BN*HD = 64*64 = 4096 halves = 512 chunks of 8 halves (16B).
    // NTH=128 < 512, so each thread handles 512/128 = 4 chunks (rows).
    // Thread tid loads rows: tid/8, tid/8+16, tid/8+32, tid/8+48 of K (and V).
    // col = (tid%8)*8, which is 16B aligned. ✓
#define LOAD_KV_SYNC(tile_n_) do { \
    { \
        const int _cs = (tile_n_) * BN; \
        const int _d  = (tid % (HD / 8)) * 8; \
        for (int _roff = 0; _roff < BN; _roff += NTH / (HD / 8)) { \
            const int _r  = (tid / (HD / 8)) + _roff; \
            const int _gc = _cs + _r; \
            if (_r < BN) { \
                if (_gc < seq_len) { \
                    cp_async_16B(&smem_k[_r][_d], &Kbh[_gc * HD + _d]); \
                    cp_async_16B(&smem_v[_r][_d], &Vbh[_gc * HD + _d]); \
                } else { \
                    smem_k[_r][_d+0] = __float2half(0.f); \
                    smem_k[_r][_d+1] = __float2half(0.f); \
                    smem_k[_r][_d+2] = __float2half(0.f); \
                    smem_k[_r][_d+3] = __float2half(0.f); \
                    smem_k[_r][_d+4] = __float2half(0.f); \
                    smem_k[_r][_d+5] = __float2half(0.f); \
                    smem_k[_r][_d+6] = __float2half(0.f); \
                    smem_k[_r][_d+7] = __float2half(0.f); \
                    smem_v[_r][_d+0] = __float2half(0.f); \
                    smem_v[_r][_d+1] = __float2half(0.f); \
                    smem_v[_r][_d+2] = __float2half(0.f); \
                    smem_v[_r][_d+3] = __float2half(0.f); \
                    smem_v[_r][_d+4] = __float2half(0.f); \
                    smem_v[_r][_d+5] = __float2half(0.f); \
                    smem_v[_r][_d+6] = __float2half(0.f); \
                    smem_v[_r][_d+7] = __float2half(0.f); \
                } \
            } \
        } \
        cp_async_commit(); \
    } \
} while(0)

    __syncthreads();  // Q load complete

    // ---- Main loop (BN=64: 8 iterations for N=512 vs 32 before) ----
    for (int tn = 0; tn < ntiles; tn++) {
        const int cs = tn * BN;

        // Load K and V tile (4 rows per thread)
        LOAD_KV_SYNC(tn);

        // ---- Preload mask to smem + sparse block skip check ----
        // 128 threads cooperatively load BROWS=64 rows × 2 uint32 words = 128 words.
        // Each thread loads 1 word (128 threads / 128 words = 1 word/thread).
        {
            const int mi = tid;  // tid in [0, 128), load word mi
            const int mrow = mi / 2;   // row index in [0, 64)
            const int mwi  = mi % 2;   // word index: 0=low 32 bits, 1=high 32 bits
            const int gr_m = blk_row + mrow;
            uint32_t mval = 0u;
            if (gr_m < seq_len) {
                const int col_base = cs + mwi * 32;
                if (col_base < seq_len) {
                    int wi = col_base / 32, bo = col_base % 32;
                    mval = Mbh[gr_m * nw + wi] >> bo;
                    if (bo > 0 && (wi + 1) < nw)
                        mval |= Mbh[gr_m * nw + wi + 1] << (32 - bo);
                    int vc = min(32, seq_len - col_base);
                    if (vc < 32) mval &= (1u << vc) - 1u;
                }
            }
            smem_mask[mrow][mwi] = mval;
        }

        // Sparse skip: check if any row in this warp has any unmasked bit
        // Use smem_mask that was just written (need syncthreads after K/V load anyway)
        cp_async_wait<0>();
        __syncthreads();

        // Ballot from smem_mask: each warp checks its BM=16 rows
        {
            uint32_t any_bits = 0u;
            if (lane < BM && wr + lane < seq_len) {
                any_bits = smem_mask[wid * BM + lane][0] | smem_mask[wid * BM + lane][1];
            }
            uint32_t ballot = __ballot_sync(0xffffffff, any_bits != 0u);
            if (lane == 0) warp_ballots[wid] = ballot;
        }
        __syncwarp();

        if ((warp_ballots[0] | warp_ballots[1] | warp_ballots[2] | warp_ballots[3]) == 0u) continue;

        // ---- WMMA QK^T + in-register softmax + rescale ----
        // Uses fragment layout: row_a = lane/4, row_b = lane/4+8, 4 lanes per row.
        // Each lane holds 16 scores per row (4 tiles × 4 elements).
        // Cross-lane reduction via __shfl_xor_sync with offsets 1,2.
        if (wr < seq_len) {
            wmma::fragment<wmma::accumulator, BM, WK, WK, float> qk_sub[NQK];
#pragma unroll
            for (int f = 0; f < NQK; f++) wmma::fill_fragment(qk_sub[f], 0.f);

#pragma unroll
            for (int kk = 0; kk < HD; kk += WK) {
                wmma::fragment<wmma::matrix_a, BM, WK, WK, __half, wmma::row_major> a;
                wmma::load_matrix_sync(a, &smem_q[wid][0][kk], HD + PP);
#pragma unroll
                for (int f = 0; f < NQK; f++) {
                    wmma::fragment<wmma::matrix_b, WK, WK, WK, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(b, &smem_k[f * WK][kk], HD + PP);
                    wmma::mma_sync(qk_sub[f], a, b, qk_sub[f]);
                }
            }

            // Apply mask + scale in-register on fragment elements
#pragma unroll
            for (int f = 0; f < NQK; f++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    int fr  = lane / 4 + ((i & 2) ? 8 : 0);
                    int fc  = f * WK + (lane % 4) * 2 + wmma_col_off[i];
                    int gr2 = wr + fr;
                    bool ok = (gr2 < seq_len) && (cs + fc < seq_len) &&
                              ((smem_mask[wid * BM + fr][fc / 32] >> (fc % 32)) & 1u);
                    qk_sub[f].x[i] = ok ? (qk_sub[f].x[i] * scale) : -65504.f;
                }
            }

            // ---- In-register softmax: find max per row ----
            float local_max_a = -65504.f, local_max_b = -65504.f;
#pragma unroll
            for (int f = 0; f < NQK; f++) {
                local_max_a = fmaxf(local_max_a, fmaxf(fmaxf(qk_sub[f].x[0], qk_sub[f].x[1]),
                                                         fmaxf(qk_sub[f].x[4], qk_sub[f].x[5])));
                local_max_b = fmaxf(local_max_b, fmaxf(fmaxf(qk_sub[f].x[2], qk_sub[f].x[3]),
                                                         fmaxf(qk_sub[f].x[6], qk_sub[f].x[7])));
            }
            local_max_a = fmaxf(local_max_a, __shfl_xor_sync(0xffffffff, local_max_a, 1));
            local_max_a = fmaxf(local_max_a, __shfl_xor_sync(0xffffffff, local_max_a, 2));
            local_max_b = fmaxf(local_max_b, __shfl_xor_sync(0xffffffff, local_max_b, 1));
            local_max_b = fmaxf(local_max_b, __shfl_xor_sync(0xffffffff, local_max_b, 2));

            float nm_a = fmaxf(local_max_a, old_max_a);
            float nm_b = fmaxf(local_max_b, old_max_b);
            float rsc_a = (old_max_a > -65000.f) ? expf(old_max_a - nm_a) : 1.f;
            float rsc_b = (old_max_b > -65000.f) ? expf(old_max_b - nm_b) : 1.f;

            // ---- Rescale acc_frag before accumulating new tile ----
#pragma unroll
            for (int f = 0; f < NF; f++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    acc_frag[f].x[i] *= ((i & 2) ? rsc_b : rsc_a);
                }
            }

            // ---- exp + local sum ----
            float local_sum_a = 0.f, local_sum_b = 0.f;
#pragma unroll
            for (int f = 0; f < NQK; f++) {
                qk_sub[f].x[0] = (qk_sub[f].x[0] > -65000.f) ? expf(qk_sub[f].x[0] - nm_a) : 0.f;
                qk_sub[f].x[1] = (qk_sub[f].x[1] > -65000.f) ? expf(qk_sub[f].x[1] - nm_a) : 0.f;
                qk_sub[f].x[4] = (qk_sub[f].x[4] > -65000.f) ? expf(qk_sub[f].x[4] - nm_a) : 0.f;
                qk_sub[f].x[5] = (qk_sub[f].x[5] > -65000.f) ? expf(qk_sub[f].x[5] - nm_a) : 0.f;
                local_sum_a += qk_sub[f].x[0] + qk_sub[f].x[1] + qk_sub[f].x[4] + qk_sub[f].x[5];

                qk_sub[f].x[2] = (qk_sub[f].x[2] > -65000.f) ? expf(qk_sub[f].x[2] - nm_b) : 0.f;
                qk_sub[f].x[3] = (qk_sub[f].x[3] > -65000.f) ? expf(qk_sub[f].x[3] - nm_b) : 0.f;
                qk_sub[f].x[6] = (qk_sub[f].x[6] > -65000.f) ? expf(qk_sub[f].x[6] - nm_b) : 0.f;
                qk_sub[f].x[7] = (qk_sub[f].x[7] > -65000.f) ? expf(qk_sub[f].x[7] - nm_b) : 0.f;
                local_sum_b += qk_sub[f].x[2] + qk_sub[f].x[3] + qk_sub[f].x[6] + qk_sub[f].x[7];
            }

            // Cross-lane sum reduction (4 lanes per row, xor 1 then 2)
            local_sum_a += __shfl_xor_sync(0xffffffff, local_sum_a, 1);
            local_sum_a += __shfl_xor_sync(0xffffffff, local_sum_a, 2);
            local_sum_b += __shfl_xor_sync(0xffffffff, local_sum_b, 1);
            local_sum_b += __shfl_xor_sync(0xffffffff, local_sum_b, 2);

            old_sum_a = old_sum_a * rsc_a + local_sum_a;
            old_sum_b = old_sum_b * rsc_b + local_sum_b;
            old_max_a = nm_a;
            old_max_b = nm_b;

            // ---- Write post-softmax P to smem_p for PV WMMA ----
#pragma unroll
            for (int f = 0; f < NQK; f++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    int fr = lane / 4 + ((i & 2) ? 8 : 0);
                    int fc = f * WK + (lane % 4) * 2 + wmma_col_off[i];
                    smem_p[wid][fr][fc] = __float2half(qk_sub[f].x[i]);
                }
            }
        }
        __syncwarp();

        // ---- WMMA PV: P[BM×BN=64] × V[BN=64×HD=64] → acc_frag[NF] ----
        if (wr < seq_len) {
#pragma unroll
            for (int f = 0; f < NF; f++) {
#pragma unroll
                for (int k = 0; k < NQK; k++) {
                    wmma::fragment<wmma::matrix_a, BM, WK, WK, __half, wmma::row_major> pa;
                    wmma::fragment<wmma::matrix_b, WK, WK, WK, __half, wmma::row_major> vb;
                    wmma::load_matrix_sync(pa, &smem_p[wid][0][k * WK], BN + PP);
                    wmma::load_matrix_sync(vb, &smem_v[k * WK][f * WK], HD + PP);
                    wmma::mma_sync(acc_frag[f], pa, vb, acc_frag[f]);
                }
            }
        }
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
                int gc2  = f * WK + fc;
                if (gr2 < seq_len) {
                    float s = ((i & 2) ? old_sum_b : old_sum_a);
                    float out_val = (s > 0.f) ? (acc_frag[f].x[i] / s) : 0.f;
                    Obh[gr2 * HD + gc2] = __float2half(out_val);
                }
                if (save_lse && LSE && f == 0 && (i == 0 || i == 2)) {
                    int fc_lse = (lane % 4) * 2 + wmma_col_off[i];
                    if (fc_lse == 0 && gr2 < seq_len) {
                        int li = (batch_id * num_heads + head_id) * seq_len + gr2;
                        float m = ((i & 2) ? old_max_b : old_max_a);
                        float s = ((i & 2) ? old_sum_b : old_sum_a);
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
