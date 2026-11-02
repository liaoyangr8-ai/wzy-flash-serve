// wzy_prefill_tc.cuh -- Unlocked 247-TFLOPS Tensor-Core (`wmma::mma_sync` HMMA.1688.F32) Prefill Engine (`PF_MAX = 8192`)
//   1. Fused NVFP4 -> Shared-Memory FP16 -> `wmma::mma_sync(16,16,16,float)` Batched MoE:
//      - 0 bytes of `w_fp16` global memory traffic!
//      - 64x shared-memory reuse of X, 32x shared-memory reuse of W!
//      - Fused Gate + Up + SwiGLU -> FP16 inside `pf_moe_gu_wmma_kernel` (ONLY 2 kernel launches per wave of experts!)
//   2. Dynamic `active_max_b` sizing (borrows ~900 MiB = only ~340 tail slots at 8K tokens!)
//   3. 48-slot Double-Buffered Async H2D DMA Ring overlapping PCIe Gen3 x16 cold-expert streaming with Tensor Cores!
//   4. GQA-12 Online Flash-Softmax Sparse QSA Attention (`12x` less KV bandwidth up to 262,144 context!)
#pragma once
#include "wzy_stage_v2.cuh"
#include "wzy_qsa_wmma.cuh"
#include <mma.h>

static constexpr int PF_MAX = 25600;       // 25.6K chunk size -> 5 chunks at 120K, 11 chunks at 256K!
static constexpr int PF_DMA_WINS = 4;      // 4 circular ring windows
static constexpr int PF_WIN_SLOTS = 128;   // 128 cold experts per window
static constexpr int PF_DMA_SLOTS = PF_DMA_WINS * PF_WIN_SLOTS; // 512 slots (1.41 GB continuous cross-layer ring!)
static constexpr int PF_TILE_M = 64;       // 64 tokens per Tensor-Core WMMA tile
static constexpr int PF_MAX_TILES = (PF_MAX * TOPK) / PF_TILE_M + NEXP + 64; // ~3776 tiles max

// ==================================================================
// PREFILL HELPER KERNELS
// ==================================================================
__global__ void silu_div_hc_to_fp16_kernel(const float* __restrict__ lo_in, __half* __restrict__ lo_fp16, int n, float inv_hc) {
    const int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 >= n) return;
    const float v0 = lo_in[idx2] * inv_hc;
    const float s0 = v0 / (1.0f + __expf(-v0));
    float s1 = 0.f;
    if (idx2 + 1 < n) {
        const float v1 = lo_in[idx2 + 1] * inv_hc;
        s1 = v1 / (1.0f + __expf(-v1));
    }
    *(__half2*) (lo_fp16 + idx2) = __floats2half2_rn(s0, s1);
}

__global__ void gr_up_gate_mean_pf_kernel(const float* __restrict__ xn, const float* __restrict__ gg,
                                          float* __restrict__ mixed, int T, int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * n_embd) return;
    const int t = idx / n_embd;
    const int d = idx - t * n_embd;
    const size_t base = (size_t) t * 4 * n_embd + d;
    float m = 0.f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float x = xn[base + (size_t) c * n_embd];
        const float g = gg[base + (size_t) c * n_embd];
        m += x / (1.0f + __expf(-g));
    }
    mixed[idx] = m * 0.25f;
}

__global__ void gdn_ab_act_pf_kernel(const float* __restrict__ ab_interleaved,
                                     const float* __restrict__ dtb, const float* __restrict__ ssm_a,
                                     float* __restrict__ decay, float* __restrict__ bt, int T, int hv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * hv) return;
    const int t = idx / hv;
    const int r = idx - t * hv;
    const float av = ab_interleaved[(size_t) t * (2 * hv) + r];
    const float bv = ab_interleaved[(size_t) t * (2 * hv) + hv + r];
    bt[idx] = 1.0f / (1.0f + __expf(-bv));
    const float v = av + dtb[r];
    const float sp = v > 20.0f ? v : log1pf(__expf(v));
    decay[idx] = __expf(sp * ssm_a[r]);
}

__global__ void pf_silu_mul_interleaved_to_fp16(const float* __restrict__ gu_in, __half* __restrict__ out_fp16, int T, int nff) {
    const size_t idx2 = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 >= (size_t) T * nff) return;
    const int t = (int) (idx2 / nff);
    const int c = (int) (idx2 - (size_t) t * nff);
    const float* row = gu_in + (size_t) t * (2 * nff);
    const float g0 = row[c],     u0 = row[nff + c];
    const float g1 = row[c + 1], u1 = row[nff + c + 1];
    const float a0 = (g0 / (1.0f + __expf(-g0))) * u0;
    const float a1 = (g1 / (1.0f + __expf(-g1))) * u1;
    *(__half2*) (out_fp16 + idx2) = __floats2half2_rn(a0, a1);
}

// ==================================================================
// FUSED NVFP4 -> SHARED FP16 -> UNLOCKED SM75 WMMA (`HMMA.1688.F32`) MOE KERNELS
// ==================================================================
struct alignas(16) PfExpMeta {
    int cnt;
    int off;
    int slot;
    int pad;
};

struct alignas(16) PfMoeTile {
    int blob_slot; // slot index in `blob_base` (`s.xblobs` for resident wave, `dma_ring` for cold wave!)
    int b_off;     // starting offset in `btok_pf` / `bwt_pf`
    int wave_row;  // contiguous intermediate row offset `tile_idx * PF_TILE_M` in `ffact_fp16`!
    int cnt;       // number of valid tokens in this tile (`1 <= cnt <= 32`)
};

__global__ void pf_moe_count_kernel(const int* __restrict__ sids, int total_k, int* __restrict__ exp_cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_k) atomicAdd(&exp_cnt[sids[i]], 1);
}

__global__ void pf_moe_scan_meta_kernel(const int* __restrict__ exp_cnt, int* __restrict__ exp_cur,
                                        const int* __restrict__ d_lut_layer, PfExpMeta* __restrict__ h_meta_uva) {
    __shared__ int s_cnt[512];
    const int tid = threadIdx.x;
    const int c = exp_cnt[tid];
    s_cnt[tid] = c;
    __syncthreads();

    int val = c;
    #pragma unroll
    for (int d = 1; d < 512; d <<= 1) {
        const int add = (tid >= d) ? s_cnt[tid - d] : 0;
        __syncthreads();
        val += add;
        s_cnt[tid] = val;
        __syncthreads();
    }
    const int excl = val - c;
    exp_cur[tid] = excl;

    PfExpMeta m;
    m.cnt = c;
    m.off = excl;
    m.slot = d_lut_layer[tid];
    m.pad = 0;
    h_meta_uva[tid] = m;
}

__global__ void pf_moe_scatter_buckets_kernel(const int* __restrict__ sids, const float* __restrict__ swts,
                                              int total_k, int* __restrict__ exp_cur,
                                              int* __restrict__ b_tok, float* __restrict__ b_wt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_k) return;
    const int eid = sids[i];
    const int dst = atomicAdd(&exp_cur[eid], 1);
    b_tok[dst] = i / TOPK;
    b_wt[dst] = swts[i];
}
// GPU-native tile builder for VRAM-resident experts (`ZERO` host synchronization!)
__global__ void pf_build_res_tiles_kernel(
    const PfExpMeta* __restrict__ meta,
    PfMoeTile* __restrict__ tiles_out,
    int* __restrict__ n_tiles_out,
    long long* __restrict__ vram_hits_accum) {
    __shared__ int s_nt[512];
    __shared__ int s_hit[512];
    const int e = threadIdx.x; // 0..511
    const PfExpMeta m = meta[e];
    const bool is_res = (m.cnt > 0 && m.slot >= 0);
    const int my_nt = is_res ? ((m.cnt + PF_TILE_M - 1) / PF_TILE_M) : 0;
    s_nt[e] = my_nt;
    s_hit[e] = is_res ? m.cnt : 0;
    __syncthreads();

    int v_nt = my_nt, v_hit = s_hit[e];
    #pragma unroll
    for (int d = 1; d < 512; d <<= 1) {
        const int a_nt = (e >= d) ? s_nt[e - d] : 0;
        const int a_hit = (e >= d) ? s_hit[e - d] : 0;
        __syncthreads();
        v_nt += a_nt;
        v_hit += a_hit;
        s_nt[e] = v_nt;
        s_hit[e] = v_hit;
        __syncthreads();
    }
    if (e == 511) {
        *n_tiles_out = v_nt;
        if (vram_hits_accum) *vram_hits_accum += (long long) v_hit;
    }
    if (my_nt > 0) {
        const int t_base = v_nt - my_nt;
        for (int t = 0; t < my_nt; ++t) {
            const int sub = t * PF_TILE_M;
            tiles_out[t_base + t] = {m.slot, m.off + sub, (t_base + t) * PF_TILE_M, min(PF_TILE_M, m.cnt - sub)};
        }
    }
}

// GPU-native tile builder for a streamed wave `[c0 .. c1)` of non-resident experts (`m.slot < 0`) in `dma_ring`!
__global__ void pf_build_cold_wave_tiles_kernel(
    const PfExpMeta* __restrict__ meta,
    int c0, int c1, int w_base,
    PfMoeTile* __restrict__ tiles_out,
    int* __restrict__ n_tiles_out) {
    __shared__ int s_scan[512];
    const int e = threadIdx.x; // 0..511
    const PfExpMeta m = meta[e];
    const int is_cold = (m.slot < 0) ? 1 : 0;
    s_scan[e] = is_cold;
    __syncthreads();

    int c_inc = is_cold;
    #pragma unroll
    for (int d = 1; d < 512; d <<= 1) {
        const int a = (e >= d) ? s_scan[e - d] : 0;
        __syncthreads();
        c_inc += a;
        s_scan[e] = c_inc;
        __syncthreads();
    }
    const int cold_idx = c_inc - is_cold;
    const bool in_wave = (is_cold != 0 && cold_idx >= c0 && cold_idx < c1);
    const int my_nt = (in_wave && m.cnt > 0) ? ((m.cnt + PF_TILE_M - 1) / PF_TILE_M) : 0;
    s_scan[e] = my_nt;
    __syncthreads();

    int v_nt = my_nt;
    #pragma unroll
    for (int d = 1; d < 512; d <<= 1) {
        const int a = (e >= d) ? s_scan[e - d] : 0;
        __syncthreads();
        v_nt += a;
        s_scan[e] = v_nt;
        __syncthreads();
    }
    if (e == 511) *n_tiles_out = v_nt;
    if (my_nt > 0) {
        const int rslot = w_base + (cold_idx - c0);
        const int t_base = v_nt - my_nt;
        for (int t = 0; t < my_nt; ++t) {
            const int sub = t * PF_TILE_M;
            tiles_out[t_base + t] = {rslot, m.off + sub, (t_base + t) * PF_TILE_M, min(PF_TILE_M, m.cnt - sub)};
        }
    }
}

__global__ void pf_init_shared_comb_kernel(const float* __restrict__ sshared, const float* __restrict__ sgate,
                                           float* __restrict__ comb, int T, int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * n_embd) return;
    const int t = idx / n_embd;
    const float sg = 1.0f / (1.0f + __expf(-sgate[t]));
    comb[idx] = sshared[idx] * sg;
}

__device__ __forceinline__ void dequant_nvfp4_sub16_to_sh_fp16(const unsigned char* rb, int s, float mat_sc, __half* dst16) {
    const float sc = ue4m3(__ldg(rb + s)) * mat_sc;
    const uint32_t* p32 = (const uint32_t*) (rb + 4 + s * 8);
    int lo0, lo1, hi0, hi1;
    unpack_nvfp4_sub16_reg(__ldg(p32), __ldg(p32 + 1), lo0, lo1, hi0, hi1);

    const uint32_t b1152 = 0x64806480u;
    const __half2 bias1152 = *reinterpret_cast<const __half2*>(&b1152);
    const __half2 sc2 = __float2half2_rn(sc);
    const uint32_t u0 = (uint32_t) lo0, u1 = (uint32_t) lo1, u2 = (uint32_t) hi0, u3 = (uint32_t) hi1;

    __half2* out2 = (__half2*) dst16;
    out2[0] = i8x2_scale_h2(u0,       bias1152, sc2);
    out2[1] = i8x2_scale_h2(u0 >> 16, bias1152, sc2);
    out2[2] = i8x2_scale_h2(u1,       bias1152, sc2);
    out2[3] = i8x2_scale_h2(u1 >> 16, bias1152, sc2);
    out2[4] = i8x2_scale_h2(u2,       bias1152, sc2);
    out2[5] = i8x2_scale_h2(u2 >> 16, bias1152, sc2);
    out2[6] = i8x2_scale_h2(u3,       bias1152, sc2);
    out2[7] = i8x2_scale_h2(u3 >> 16, bias1152, sc2);
}

// Fused Gate + Up + SwiGLU -> FP16 Tensor-Core WMMA Kernel!
// Grid: dim3(NFF / 16 = 40, n_tiles), Block: 256 (8 warps: `warp_m = 0..3` -> 64 tokens; `warp_n = 0` -> 16 Gate rows, `warp_n = 1` -> 16 Up rows!)
__global__ void __launch_bounds__(256) pf_moe_gu_wmma_kernel(
    const unsigned char* __restrict__ blob_base,
    const PfMoeTile* __restrict__ tiles,
    const int* __restrict__ b_tok,
    const __half* __restrict__ x_fp16_in,
    __half* __restrict__ ffact_fp16_out,
    const int* __restrict__ d_ntiles = nullptr) {

    if (d_ntiles != nullptr && (int) blockIdx.y >= __ldg(d_ntiles)) return;
    using namespace nvcuda;
    __shared__ int s_tok[64];
    __shared__ union {
        struct {
            __half X[64][72];
            __half W[32][72];
        } stage;
        struct {
            float gate[64][20];
            float up[64][20];
        } epi;
    } smem;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_m = warp >> 1; // 0..3 (64 tokens!)
    const int warp_n = warp & 1;  // 0 = Gate (16 rows), 1 = Up (16 rows)

    const PfMoeTile tl = tiles[blockIdx.y];
    const int cnt = tl.cnt;
    if (tid < 64) s_tok[tid] = (tid < cnt) ? b_tok[tl.b_off + tid] : 0;
    __syncthreads();

    const int r0 = blockIdx.x * 16; // 0 .. 624 (step 16)
    const unsigned char* blob = blob_base + (size_t) tl.blob_slot * PER;
    const float* tail = (const float*) (blob + PER - 16);
    const float sc_gate = tail[0], sc_up = tail[1];

    const int w_slot = tid >> 2; // 0..31 active (tid < 128)
    const int s_sub = tid & 3;
    const bool w_is_up = (w_slot >= 16);
    const int w_row = (tid < 128) ? (w_is_up ? (NFF + r0 + (w_slot - 16)) : (r0 + w_slot)) : 0;
    const float w_mat_sc = w_is_up ? sc_up : sc_gate;
    const unsigned char* w_row_ptr = blob + (size_t) w_row * GU_ROW;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int b = 0; b < (NE / 64); ++b) {
        #pragma unroll
        for (int xp = 0; xp < 2; ++xp) {
            const int x_idx = xp * 256 + tid;
            const int xm = x_idx >> 3;        // 0..63
            const int xk8 = (x_idx & 7) * 8;  // 0..56
            if (xm < cnt) {
                *(uint4*) (&smem.stage.X[xm][xk8]) = *(const uint4*) (x_fp16_in + (size_t) s_tok[xm] * NE + b * 64 + xk8);
            } else {
                *(uint4*) (&smem.stage.X[xm][xk8]) = make_uint4(0, 0, 0, 0);
            }
        }
        if (tid < 128) {
            dequant_nvfp4_sub16_to_sh_fp16(w_row_ptr + b * 36, s_sub, w_mat_sc, &smem.stage.W[w_slot][s_sub * 16]);
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 64; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &smem.stage.X[warp_m * 16][ks], 72);
            wmma::load_matrix_sync(b_frag, &smem.stage.W[warp_n * 16][ks], 72);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    if (warp_n == 0) {
        wmma::store_matrix_sync(&smem.epi.gate[warp_m * 16][0], acc, 20, wmma::mem_row_major);
    } else {
        wmma::store_matrix_sync(&smem.epi.up[warp_m * 16][0], acc, 20, wmma::mem_row_major);
    }
    __syncthreads();

    // 256 threads write `64 * 16 = 1024` `__half`s (`512` `__half2`s = 2 `__half2`s per thread)!
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int idx2 = pass * 256 + tid; // 0..511
        const int m = idx2 >> 3;           // 0..63
        const int c2 = (idx2 & 7) * 2;     // 0..14
        if (m < cnt) {
            const float g0 = smem.epi.gate[m][c2],     u0 = smem.epi.up[m][c2];
            const float g1 = smem.epi.gate[m][c2 + 1], u1 = smem.epi.up[m][c2 + 1];
            const float a0 = (g0 / (1.0f + __expf(-g0))) * u0;
            const float a1 = (g1 / (1.0f + __expf(-g1))) * u1;
            *(__half2*) (ffact_fp16_out + (size_t) (tl.wave_row + m) * NFF + r0 + c2) = __floats2half2_rn(a0, a1);
        }
    }
}

// Grid: dim3(NE / 32 = 80, n_tiles), Block: 256 (8 warps: `warp_m = 0..3` x `warp_n = 0..1` -> `[64, 32]` output tile!)
__global__ void __launch_bounds__(256) pf_moe_down_wmma_kernel(
    const unsigned char* __restrict__ blob_base,
    const PfMoeTile* __restrict__ tiles,
    const int* __restrict__ b_tok, const float* __restrict__ b_wt,
    const __half* __restrict__ ffact_fp16_in,
    float* __restrict__ comb_out,
    const int* __restrict__ d_ntiles = nullptr) {

    if (d_ntiles != nullptr && (int) blockIdx.y >= __ldg(d_ntiles)) return;
    using namespace nvcuda;
    __shared__ int s_tok[64];
    __shared__ float s_wt[64];
    __shared__ union {
        struct {
            __half X[64][72];
            __half W[32][72];
        } stage;
        float C[64][36];
    } smem;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_m = warp >> 1; // 0..3 (64 tokens!)
    const int warp_n = warp & 1;  // 0..1 (32 rows!)

    const PfMoeTile tl = tiles[blockIdx.y];
    const int cnt = tl.cnt;
    if (tid < 64) {
        s_tok[tid] = (tid < cnt) ? b_tok[tl.b_off + tid] : 0;
        s_wt[tid]  = (tid < cnt) ? b_wt[tl.b_off + tid] : 0.f;
    }
    __syncthreads();

    const int r0 = blockIdx.x * 32; // 0 .. 2528 (step 32)
    const unsigned char* blob = blob_base + (size_t) tl.blob_slot * PER;
    const float sc_down = ((const float*) (blob + PER - 16))[2];
    const int w_slot = tid >> 2;
    const int s_sub = tid & 3;
    const unsigned char* w_row_ptr = blob + 2ull * NFF * GU_ROW + (size_t) (r0 + (tid < 128 ? w_slot : 0)) * D_ROW;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int b = 0; b < (NFF / 64); ++b) {
        #pragma unroll
        for (int xp = 0; xp < 2; ++xp) {
            const int x_idx = xp * 256 + tid;
            const int xm = x_idx >> 3;        // 0..63
            const int xk8 = (x_idx & 7) * 8;  // 0..56
            if (xm < cnt) {
                *(uint4*) (&smem.stage.X[xm][xk8]) = *(const uint4*) (ffact_fp16_in + (size_t) (tl.wave_row + xm) * NFF + b * 64 + xk8);
            } else {
                *(uint4*) (&smem.stage.X[xm][xk8]) = make_uint4(0, 0, 0, 0);
            }
        }
        if (tid < 128) {
            dequant_nvfp4_sub16_to_sh_fp16(w_row_ptr + b * 36, s_sub, sc_down, &smem.stage.W[w_slot][s_sub * 16]);
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 64; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &smem.stage.X[warp_m * 16][ks], 72);
            wmma::load_matrix_sync(b_frag, &smem.stage.W[warp_n * 16][ks], 72);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(&smem.C[warp_m * 16][warp_n * 16], acc, 36, wmma::mem_row_major);
    __syncthreads();

    // 100% Coalesced warp-per-row scatter-add: all 32 lanes of warp `warp` write 32 contiguous floats (`r0 + lane`) of token `m`!
    const int lane = tid & 31;
    for (int m = warp; m < cnt; m += 8) {
        atomicAdd(&comb_out[(size_t) s_tok[m] * NE + r0 + lane], s_wt[m] * smem.C[m][lane]);
    }
}

// ==================================================================
// BORROWED-VRAM PREFILL ARENA & STAGE EXECUTOR
// ==================================================================
struct PrefillStageCtx {
    int dev = 0;
    cudaStream_t dma_stream = nullptr;
    cudaEvent_t ev_dma_win[2] = {nullptr, nullptr};
    cudaEvent_t ev_comp_win[2] = {nullptr, nullptr};
    cudaEvent_t ev_dma_w[PF_DMA_WINS] = {nullptr};
    cudaEvent_t ev_comp_w[PF_DMA_WINS] = {nullptr};
    cudaEvent_t ev_meta = nullptr;
    cudaEvent_t ev_in_copied = nullptr;
    PfExpMeta *h_meta = nullptr, *d_meta = nullptr;
    long long *h_vhits = nullptr, *d_vhits = nullptr;
    int *h_pstart_ring = nullptr;
    size_t pstart_cursor = 0;
    PfMoeTile *h_tile_stage = nullptr;
    size_t tile_stage_cursor = 0;
    static constexpr size_t TILE_STAGE_CAP = 65536;
    float *h_emb_pf = nullptr, *h_ple_pf = nullptr;

    bool borrowed_dirty = false;
    int saved_cap = 0;
    int active_cap = 0;
    std::vector<std::pair<int, int>> evicted_gkey_slot;

    void init(const StageGpu& s) {
        dev = s.dev;
        CHECK(cudaSetDevice(dev));
        CHECK(cudaStreamCreateWithFlags(&dma_stream, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            CHECK(cudaEventCreateWithFlags(&ev_dma_win[i], cudaEventDisableTiming));
            CHECK(cudaEventCreateWithFlags(&ev_comp_win[i], cudaEventDisableTiming));
        }
        for (int i = 0; i < PF_DMA_WINS; ++i) {
            CHECK(cudaEventCreateWithFlags(&ev_dma_w[i], cudaEventDisableTiming));
            CHECK(cudaEventCreateWithFlags(&ev_comp_w[i], cudaEventDisableTiming));
        }
        CHECK(cudaEventCreateWithFlags(&ev_meta, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&ev_in_copied, cudaEventDisableTiming));
        CHECK(cudaHostAlloc(&h_meta, NEXP * sizeof(PfExpMeta), cudaHostAllocMapped | cudaHostAllocPortable));
        CHECK(cudaHostGetDevicePointer(&d_meta, h_meta, 0));
        CHECK(cudaHostAlloc(&h_vhits, sizeof(long long), cudaHostAllocMapped | cudaHostAllocPortable));
        *h_vhits = 0;
        CHECK(cudaHostGetDevicePointer(&d_vhits, h_vhits, 0));
        CHECK(cudaHostAlloc(&h_pstart_ring, 32 * sizeof(int), cudaHostAllocPortable));
        CHECK(cudaHostAlloc(&h_tile_stage, TILE_STAGE_CAP * sizeof(PfMoeTile), cudaHostAllocPortable));
        if (s.l_begin == 0) {
            CHECK(cudaHostAlloc(&h_emb_pf, (size_t) PF_MAX * NE * sizeof(float), cudaHostAllocPortable));
        }
        if (s.has_ple) {
            CHECK(cudaHostAlloc(&h_ple_pf, (size_t) PF_MAX * NE * sizeof(float), cudaHostAllocPortable));
        }
        saved_cap = s.cap;
        active_cap = s.cap;
    }

    void sync_stats(StageGpu& s) {
        if (h_vhits && *h_vhits > 0) {
            s.total_vram_hits += *h_vhits;
            *h_vhits = 0;
        }
    }

    uint8_t* borrow_workspace(StageGpu& s, size_t need_bytes) {
        CHECK(cudaSetDevice(s.dev));
        const int need_slots = (int) ((need_bytes + PER - 1) / PER);
        const int keep_slots = std::max(64, saved_cap - need_slots);
        if (!borrowed_dirty || keep_slots < active_cap) {
            for (int gkey = s.l_begin * NEXP; gkey < s.l_end * NEXP; ++gkey) {
                const int sl = s.h_lut[gkey];
                if (sl >= keep_slots) {
                    evicted_gkey_slot.emplace_back(gkey, sl);
                    s.h_lut[gkey] = -1;
                    if (s.h_lut_pinned) s.h_lut_pinned[gkey] = -1;
                }
            }
            CHECK(cudaMemcpyAsync(s.d_lut, s.h_lut.data(), 48 * NEXP * 4, cudaMemcpyHostToDevice, s.stream));
            active_cap = keep_slots;
            borrowed_dirty = true;
        }
        return s.xblobs + (size_t) active_cap * PER;
    }

    void restore_borrowed(StageGpu& s, const uint8_t* host_pack) {
        if (!borrowed_dirty) return;
        CHECK(cudaSetDevice(s.dev));
        for (const auto& kv : evicted_gkey_slot) {
            const int gkey = kv.first, sl = kv.second;
            s.h_lut[gkey] = sl;
            if (s.h_lut_pinned) s.h_lut_pinned[gkey] = sl;
            CHECK(cudaMemcpyAsync(s.xblobs + (size_t) sl * PER,
                                  host_pack + (size_t) gkey * PER,
                                  PER, cudaMemcpyHostToDevice, s.stream));
        }
        CHECK(cudaMemcpyAsync(s.d_lut, s.h_lut.data(), 48 * NEXP * 4, cudaMemcpyHostToDevice, s.stream));
        evicted_gkey_slot.clear();
        active_cap = saved_cap;
        borrowed_dirty = false;
    }

    float* run_stage_prefill(StageGpu& s, int p_start, int B, const uint8_t* host_pack, const float* prev_stage_R_dev,
                             cudaEvent_t ev_rcopied_out = nullptr, std::atomic<bool>* p2p_queued_flag = nullptr) {
        CHECK(cudaSetDevice(s.dev));

        const int active_max_b = (p_start + B) / IDX_BLK + 4;
        const int total_k = B * TOPK;
        size_t off = 0;
        auto bump_bytes = [&](size_t b) -> size_t {
            size_t r = off;
            off = (off + b + 255ull) & ~255ull;
            return r;
        };

        const size_t max_tiles_B = (size_t) (total_k / PF_TILE_M + NEXP + 64);
        const int qsa_score_b = std::min(B, 2048);

        // 1. Live across the whole layer (disjoint from BOTH Attention and FFN union branches!)
        const size_t o_R       = bump_bytes((size_t) B * HCD * 4);
        const size_t o_block   = bump_bytes((size_t) B * NE * 4);
        const size_t o_comb    = o_block; // `block_pf` is consumed by `gr_write` before `comb_pf` is written!
        const size_t o_inj     = bump_bytes((size_t) B * HC * 4);
        const size_t o_lo      = bump_bytes((size_t) B * HCLR * 4);
        const size_t o_xfp16   = bump_bytes((size_t) B * HCD * sizeof(__half));
        const size_t o_bufA    = bump_bytes((size_t) B * HCD * 4); // holds `gg_h + lo_h` in `tc_gr_read` & `conv_h` in GDN
        const size_t o_ple_emb = s.has_ple ? bump_bytes((size_t) B * NE * 4) : 0;
        const size_t o_dma     = bump_bytes((size_t) (B >= 8192 ? PF_DMA_SLOTS : 2 * PF_WIN_SLOTS) * PER);

        const size_t union_base = off;

        // 2. Union Branch A: Attention / PLE Scratch (dead before FFN starts!)
        const size_t o_mixed   = o_block; // aliases `block_pf` (100% dead at `l==0` init and `l==1` PLE!) -> saves 251 MB/GPU!
        const size_t o_xn      = bump_bytes((size_t) B * (2 * QD) * 4);
        const size_t o_bufB    = bump_bytes((size_t) B * GV * 4);
        const size_t kv_mirror_bytes = (B > 2048 && p_start + B > KV_VRAM_RING && s.hk_q) ? ((size_t) (p_start + B - KV_VRAM_RING) * 1056ull + 4096ull) : 0ull;
        const size_t o_on      = bump_bytes(std::max((size_t) B * GV * 4, kv_mirror_bytes));
        const size_t o_kk      = bump_bytes((size_t) B * KVD * 4);
        const size_t o_vv      = bump_bytes((size_t) B * KVD * 4);
        const size_t o_idx_raw = bump_bytes((size_t) B * IDX_DIM * 4);
        const size_t o_q_idx   = bump_bytes((size_t) B * IDX_HEADS * IDX_DIM * 4);
        const size_t o_sel_ids = bump_bytes((size_t) B * IDX_TOPK * 4);
        const size_t o_sel_cnt = bump_bytes((size_t) B * 4);
        const size_t o_qsa_sc  = bump_bytes((size_t) qsa_score_b * active_max_b * 4);
        const size_t o_decay   = bump_bytes((size_t) B * VH * 4);
        const size_t o_bt      = bump_bytes((size_t) B * VH * 4);
        const size_t o_araw    = bump_bytes((size_t) B * (2 * VH) * 4);
        const size_t o_braw    = o_araw;
        const size_t o_ple_gat = o_bufA;  // aliases `bufA_pf` in PLE (after `ple_nk` is consumed!) -> saves 1.01 GB on GPU0!
        const size_t attn_end  = off;

        // 3. Union Branch B: FFN / MoE Scratch (aliases Attention scratch for ZERO extra VRAM!)
        off = union_base;
        const size_t o_mixed2  = bump_bytes((size_t) B * NE * 4);
        const size_t o_sshared = bump_bytes((size_t) B * NE * 4);
        const size_t o_slog    = bump_bytes((size_t) B * NEXP * 4);
        const size_t o_sids    = bump_bytes((size_t) total_k * 4);
        const size_t o_swts    = bump_bytes((size_t) total_k * 4);
        const size_t o_sff     = bump_bytes((size_t) B * (2 * NFF) * 4);
        const size_t o_sup     = o_sff;
        const size_t o_tmpff   = o_sff;
        const size_t o_sgate   = bump_bytes((size_t) B * 4);
        const size_t o_m2_fp16 = bump_bytes((size_t) B * NE * sizeof(__half));
        const size_t o_ff_fp16 = bump_bytes(max_tiles_B * PF_TILE_M * NFF * sizeof(__half));
        const size_t o_dtiles  = bump_bytes(max_tiles_B * sizeof(PfMoeTile));
        const size_t o_exp_cnt = bump_bytes(NEXP * 4);
        const size_t o_exp_cur = bump_bytes(NEXP * 4);
        const size_t o_btok    = bump_bytes((size_t) total_k * 4);
        const size_t o_bwt     = bump_bytes((size_t) total_k * 4);
        const size_t o_dmeta   = bump_bytes(NEXP * sizeof(PfExpMeta));
        const size_t o_ntiles  = bump_bytes(16);
        const size_t ffn_end   = off;
        uint8_t* ws = borrow_workspace(s, std::max(attn_end, ffn_end));
        auto pf_f = [&](size_t o) { return (float*) (ws + o); };
        auto pf_i = [&](size_t o) { return (int*) (ws + o); };

        float* R_pf        = pf_f(o_R);
        float* xn_pf       = pf_f(o_xn);
        float* bufA_pf     = pf_f(o_bufA);
        float* bufB_pf     = pf_f(o_bufB);
        float* on_pf       = pf_f(o_on);
        float* mixed_pf    = pf_f(o_mixed);
        float* mixed2_pf   = pf_f(o_mixed2);
        float* block_pf    = pf_f(o_block);
        float* comb_pf     = pf_f(o_comb);
        float* sshared_pf  = pf_f(o_sshared);
        float* lo_pf       = pf_f(o_lo);
        float* inj_pf      = pf_f(o_inj);
        float* kk_pf       = pf_f(o_kk);
        float* vv_pf       = pf_f(o_vv);
        float* idx_raw_pf  = pf_f(o_idx_raw);
        float* q_idx_pf    = pf_f(o_q_idx);
        int*   sel_ids_pf  = pf_i(o_sel_ids);
        int*   sel_cnt_pf  = pf_i(o_sel_cnt);
        float* qsa_sc_pf   = pf_f(o_qsa_sc);
        float* decay_pf    = pf_f(o_decay);
        float* bt_pf       = pf_f(o_bt);
        float* araw_pf     = pf_f(o_araw);
        float* braw_pf     = pf_f(o_braw);
        float* slog_pf     = pf_f(o_slog);
        int*   sids_pf     = pf_i(o_sids);
        float* swts_pf     = pf_f(o_swts);
        float* sff_pf      = pf_f(o_sff);
        float* sup_pf      = pf_f(o_sup);
        float* tmpff_pf    = pf_f(o_tmpff);
        float* sgate_pf    = pf_f(o_sgate);
        __half* xfp16_pf   = (__half*) (ws + o_xfp16);
        __half* m2_fp16_pf = (__half*) (ws + o_m2_fp16);
        __half* ff_fp16_pf = (__half*) (ws + o_ff_fp16);
        PfMoeTile* d_tiles_pf = (PfMoeTile*) (ws + o_dtiles);
        int*   exp_cnt_pf  = pf_i(o_exp_cnt);
        int*   exp_cur_pf  = pf_i(o_exp_cur);
        int*   btok_pf     = pf_i(o_btok);
        float* bwt_pf      = pf_f(o_bwt);
        uint8_t* dma_ring  = ws + o_dma;
        PfExpMeta* d_meta_pf = (PfExpMeta*) (ws + o_dmeta);
        int*   d_ntiles_pf = pf_i(o_ntiles);

        int* pstart_slot = &h_pstart_ring[(pstart_cursor++) & 31];
        *pstart_slot = p_start;
        CHECK(cudaMemcpyAsync(s.dpos, pstart_slot, 4, cudaMemcpyHostToDevice, s.stream));

        if (s.l_begin == 0) {
            CHECK(cudaMemcpyAsync(mixed_pf, h_emb_pf, (size_t) B * NE * 4, cudaMemcpyHostToDevice, s.stream));
            CHECK(cudaMemsetAsync(bufA_pf, 0, (size_t) B * HCD * 4, s.stream));
            mtp_add_streams_bcast<<<((size_t) B * HCD + 255) / 256, 256, 0, s.stream>>>(bufA_pf, mixed_pf, R_pf, NE, HC, B);
        } else {
            CHECK(cudaMemcpyPeerAsync(R_pf, s.dev, prev_stage_R_dev, 0, (size_t) B * HCD * 4, s.stream));
            if (ev_rcopied_out) CHECK(cudaEventRecord(ev_rcopied_out, s.stream));
            if (p2p_queued_flag) p2p_queued_flag->store(true, std::memory_order_release);
        }
        if (s.has_ple) {
            CHECK(cudaMemcpyAsync(pf_f(o_ple_emb), h_ple_pf, (size_t) B * NE * 4, cudaMemcpyHostToDevice, s.stream));
        }
        if (s.l_begin == 0) CHECK(cudaEventRecord(ev_in_copied, s.stream));

        const float one = 1.0f, zero = 0.0f;
        auto tc_gemm_fp16 = [&](int rows, int cols, int M, const __half* x_h, float* y) {
            cublasGemmEx(s.cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, M, cols, &one,
                         s.w_fp16, CUDA_R_16F, cols, x_h, CUDA_R_16F, cols,
                         &zero, y, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        auto tc_bf16 = [&](const Bf& W, const float* x, float* y, int rows, int cols) {
            const size_t tot = (size_t) rows * cols;
            bf16_to_fp16_vec<<<((tot + 1) / 2 + 255) / 256, 256, 0, s.stream>>>(W.p, s.w_fp16, tot);
            f32_to_fp16_vec<<<(((size_t) B * cols + 1) / 2 + 255) / 256, 256, 0, s.stream>>>(x, xfp16_pf, (size_t) B * cols);
            tc_gemm_fp16(rows, cols, B, xfp16_pf, y);
        };
        auto tc_gr_read = [&](const Dev& gn, const Q8& gd, const Q8& gu, const Bf& gi,
                              const float* bo_prev, const float* inj_prev, float* mx, __half* mx_h, float* inj_out) {
            __half* xn_h = xfp16_pf;
            __half* gg_h = (__half*) bufA_pf;
            __half* lo_h = ((__half*) bufA_pf) + (size_t) B * HCD;
            pf_gr_norm_fp16<<<dim3(HC, B), 256, 0, s.stream>>>(R_pf, gn.p, bo_prev, inj_prev, xn_h, NE, HC, EPS);
            if (gi.p && inj_out) {
                const size_t itot = (size_t) HC * HCD;
                bf16_to_fp16_vec<<<((itot + 1) / 2 + 255) / 256, 256, 0, s.stream>>>(gi.p, s.w_fp16, itot);
                tc_gemm_fp16(HC, HCD, B, xn_h, inj_out);
            }
            const size_t dtot = (size_t) HCLR * HCD;
            dequant_q8p_to_fp16<<<((dtot + 3) / 4 + 255) / 256, 256, 0, s.stream>>>(gd.p, gd.s, s.w_fp16, HCLR, HCD);
            tc_gemm_fp16(HCLR, HCD, B, xn_h, lo_pf);
            silu_div_hc_to_fp16_kernel<<<(((size_t) B * HCLR + 1) / 2 + 255) / 256, 256, 0, s.stream>>>(lo_pf, lo_h, B * HCLR, 1.0f / (float) HC);
            const size_t utot = (size_t) HCD * HCLR;
            dequant_q8p_to_fp16<<<((utot + 3) / 4 + 255) / 256, 256, 0, s.stream>>>(gu.p, gu.s, s.w_fp16, HCD, HCLR);
            cublasGemmEx(s.cublas, CUBLAS_OP_T, CUBLAS_OP_N, HCD, B, HCLR, &one,
                         s.w_fp16, CUDA_R_16F, HCLR, lo_h, CUDA_R_16F, HCLR,
                         &zero, gg_h, CUDA_R_16F, HCD, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            pf_gr_up_mean_fp16<<<((size_t) B * NE + 255) / 256, 256, 0, s.stream>>>(xn_h, gg_h, mx, mx_h, B, NE);
        };

        // Launch a wave of `n_tiles` Tensor-Core WMMA tiles in JUST 2 KERNEL LAUNCHES!
        auto launch_wmma_wave = [&](const uint8_t* blob_base, const std::vector<PfMoeTile>& wave_tiles) {
            if (wave_tiles.empty()) return;
            const size_t n_tiles = wave_tiles.size();
            if (tile_stage_cursor + n_tiles > TILE_STAGE_CAP) tile_stage_cursor = 0;
            PfMoeTile* h_src = h_tile_stage + tile_stage_cursor;
            tile_stage_cursor += n_tiles;
            std::memcpy(h_src, wave_tiles.data(), n_tiles * sizeof(PfMoeTile));
            CHECK(cudaMemcpyAsync(d_tiles_pf, h_src, n_tiles * sizeof(PfMoeTile), cudaMemcpyHostToDevice, s.stream));

            pf_moe_gu_wmma_kernel<<<dim3(NFF / 16, (unsigned) n_tiles), 256, 0, s.stream>>>(
                blob_base, d_tiles_pf, btok_pf, m2_fp16_pf, ff_fp16_pf);
            pf_moe_down_wmma_kernel<<<dim3(NE / 32, (unsigned) n_tiles), 256, 0, s.stream>>>(
                blob_base, d_tiles_pf, btok_pf, bwt_pf, ff_fp16_pf, comb_pf);
        };

        const float* pending_bo = nullptr;
        const float* pending_inj = nullptr;

        const int prof_lvl = std::getenv("WZY_PF_PROF") ? std::atoi(std::getenv("WZY_PF_PROF")) : 0;
        const bool prof_on = (prof_lvl > 0);
        double ms_gdn = 0, ms_qsa = 0, ms_sh = 0, ms_res = 0, ms_cold = 0;
        long long tot_cold_blobs = 0;
        auto sync_tick = [&]() {
            if (prof_on) cudaStreamSynchronize(s.stream);
            return std::chrono::steady_clock::now();
        };

        const bool stream_all_ring = (B >= 8192);
        struct ColdWavePlan {
            int l;
            int c0, c1;
            int eids[PF_WIN_SLOTS];
        };
        std::vector<ColdWavePlan> ring_waves;
        std::vector<int> layer_w0(49, 0), layer_w1(49, 0), layer_nres(49, 0);
        if (stream_all_ring) {
            ring_waves.reserve(96);
            for (int l = s.l_begin; l < s.l_end; ++l) {
                layer_w0[l] = (int) ring_waves.size();
                int n_res_l = 0;
                int cold_list[NEXP], nc = 0;
                for (int e = 0; e < NEXP; ++e) {
                    if (s.h_lut[l * NEXP + e] >= 0) ++n_res_l;
                    else cold_list[nc++] = e;
                }
                layer_nres[l] = n_res_l;
                for (int c0 = 0; c0 < nc; c0 += PF_WIN_SLOTS) {
                    ColdWavePlan wp;
                    wp.l = l;
                    wp.c0 = c0;
                    wp.c1 = std::min(nc, c0 + PF_WIN_SLOTS);
                    for (int k = c0; k < wp.c1; ++k) wp.eids[k - c0] = cold_list[k];
                    ring_waves.push_back(wp);
                }
                layer_w1[l] = (int) ring_waves.size();
                tot_cold_blobs += nc;
            }
            for (int w = 0; w < PF_DMA_WINS; ++w) CHECK(cudaEventRecord(ev_comp_w[w], s.stream));
        } else {
            CHECK(cudaEventRecord(ev_comp_win[0], s.stream));
            CHECK(cudaEventRecord(ev_comp_win[1], s.stream));
        }

        auto enqueue_ring_wave = [&](int wid) {
            if (wid >= (int) ring_waves.size()) return;
            const int win = wid & (PF_DMA_WINS - 1);
            const int w_base = win * PF_WIN_SLOTS;
            const ColdWavePlan& wp = ring_waves[wid];
            const int nw = wp.c1 - wp.c0;
            CHECK(cudaStreamWaitEvent(dma_stream, ev_comp_w[win], 0));
            int i = 0;
            while (i < nw) {
                int j = i + 1;
                while (j < nw && wp.eids[j] == wp.eids[j - 1] + 1) ++j;
                const int gkey0 = wp.l * NEXP + wp.eids[i];
                CHECK(cudaMemcpyAsync(dma_ring + (size_t) (w_base + i) * PER,
                                      host_pack + (size_t) gkey0 * PER,
                                      (size_t) (j - i) * PER, cudaMemcpyHostToDevice, dma_stream));
                i = j;
            }
            CHECK(cudaEventRecord(ev_dma_w[win], dma_stream));
        };

        if (stream_all_ring) {
            // Prime all 4 windows (512 cold experts = ~1.5+ full layers ahead!) BEFORE Layer 0's Attention starts!
            for (int w = 0; w < PF_DMA_WINS; ++w) enqueue_ring_wave(w);
        }

        for (int l = s.l_begin; l < s.l_end; ++l) {
            auto t_l0 = sync_tick();

            LayerW& w = s.L[l];
            if (l == 1 && s.has_ple) {
                if (pending_bo) {
                    gr_write_multi<<<dim3(80, B), 256, 0, s.stream>>>(R_pf, pending_bo, pending_inj, NE, HC);
                    pending_bo = nullptr; pending_inj = nullptr;
                }
                float* ple_emb = pf_f(o_ple_emb);
                float* ple_gat = pf_f(o_ple_gat);
                ple_group_multi<<<dim3(HC, B), 256, 0, s.stream>>>(R_pf, s.ple_nq.p, xn_pf, NE, HC, EPS);
                tc_bf16(s.ple_wk, ple_emb, bufA_pf, HCD, NE);
                ple_group_multi<<<dim3(HC, B), 256, 0, s.stream>>>(bufA_pf, s.ple_nk.p, bufA_pf, NE, HC, EPS);
                tc_bf16(s.ple_wv, ple_emb, mixed_pf, NE, NE);
                ple_gate_gv_multi<<<dim3(HC, B), 256, 0, s.stream>>>(bufA_pf, xn_pf, mixed_pf, ple_gat, HC, NE);
                ple_group_multi<<<dim3(HC, B), 256, 0, s.stream>>>(ple_gat, s.ple_nc.p, xn_pf, NE, HC, EPS);
                ple_conv_multi<<<(HCD + 255) / 256, 256, 0, s.stream>>>(
                    xn_pf, s.ple_cv.p, s.ple_hist.p, ple_gat, R_pf, HCD, PLE_NG, PLE_KERN, B, nullptr);
            }

            tc_gr_read(w.hc_norm[0], w.hc_down[0], w.hc_up[0], w.hc_inj[0], pending_bo, pending_inj, nullptr, xfp16_pf, inj_pf);

            if (!w.qsa) {
                dequant_q8p_to_fp16<<<((size_t) QKV * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.qkv.p, w.qkv.s, s.w_fp16, QKV, NE);
                tc_gemm_fp16(QKV, NE, B, xfp16_pf, xn_pf);
                dequant_q8p_to_fp16<<<((size_t) GV * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.wgate.p, w.wgate.s, s.w_fp16, GV, NE);
                tc_gemm_fp16(GV, NE, B, xfp16_pf, bufB_pf);

                bf16_to_fp16_vec<<<((size_t) VH * NE / 2 + 255) / 256, 256, 0, s.stream>>>(w.alpha.p, s.w_fp16, (size_t) VH * NE);
                bf16_to_fp16_vec<<<((size_t) VH * NE / 2 + 255) / 256, 256, 0, s.stream>>>(w.beta.p, s.w_fp16 + (size_t) VH * NE, (size_t) VH * NE);
                tc_gemm_fp16(2 * VH, NE, B, xfp16_pf, araw_pf);
                gdn_ab_act_pf_kernel<<<((size_t) B * VH + 255) / 256, 256, 0, s.stream>>>(
                    araw_pf, w.dtb.p, w.ssm_a.p, decay_pf, bt_pf, B, VH);

                pf_gdn_conv_parallel<<<dim3(QKV / 128, B), 128, 0, s.stream>>>(
                    s.conv[l].p, xn_pf, w.conv1d.p, bufA_pf, QKV, 2 * GH, EPS, B);
                pf_gdn_conv_update_state<<<(QKV + 255) / 256, 256, 0, s.stream>>>(
                    s.conv[l].p, xn_pf, QKV, B);
                pf_gdn_recur_kh_kernel<<<64, dim3(32, 4), 0, s.stream>>>(
                    s.ssm[l].p, bufA_pf, decay_pf, bt_pf, xn_pf, B);
                pf_gdn_rmsnorm_gate_parallel<<<dim3(VH, B), 128, 0, s.stream>>>(
                    xn_pf, bufB_pf, w.ssm_norm.p, xfp16_pf, VH, EPS);
                dequant_q8p_to_fp16<<<((size_t) NE * GV / 4 + 255) / 256, 256, 0, s.stream>>>(w.wout.p, w.wout.s, s.w_fp16, NE, GV);
                tc_gemm_fp16(NE, GV, B, xfp16_pf, block_pf);
            } else {
                const int ql = w.qsa_local;
                dequant_q8p_to_fp16<<<((size_t) (2 * QD) * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.wq.p, w.wq.s, s.w_fp16, 2 * QD, NE);
                tc_gemm_fp16(2 * QD, NE, B, xfp16_pf, xn_pf);
                dequant_q8p_to_fp16<<<((size_t) KVD * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.wk.p, w.wk.s, s.w_fp16, KVD, NE);
                tc_gemm_fp16(KVD, NE, B, xfp16_pf, kk_pf);
                dequant_q8p_to_fp16<<<((size_t) KVD * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.wv.p, w.wv.s, s.w_fp16, KVD, NE);
                tc_gemm_fp16(KVD, NE, B, xfp16_pf, vv_pf);
                bf16_to_fp16_vec<<<((size_t) IDX_DIM * NE / 2 + 255) / 256, 256, 0, s.stream>>>(w.idx_k.p, s.w_fp16, (size_t) IDX_DIM * NE);
                tc_gemm_fp16(IDX_DIM, NE, B, xfp16_pf, idx_raw_pf);
                bf16_to_fp16_vec<<<((size_t) IDX_HEADS * IDX_DIM * NE / 2 + 255) / 256, 256, 0, s.stream>>>(w.idx_q.p, s.w_fp16, (size_t) IDX_HEADS * IDX_DIM * NE);
                tc_gemm_fp16(IDX_HEADS * IDX_DIM, NE, B, xfp16_pf, q_idx_pf);

                fused_head_norm_rope_multi<<<dim3(QH, B), 128, 0, s.stream>>>(
                    xn_pf, w.q_norm.p, s.cos_tab.p, s.sin_tab.p, s.dpos, QH, HD, 2 * HD, 2 * QD, EPS);
                fused_head_norm_rope_multi<<<dim3(KVH, B), 128, 0, s.stream>>>(
                    kk_pf, w.k_norm.p, s.cos_tab.p, s.sin_tab.p, s.dpos, KVH, HD, HD, KVD, EPS);
                fused_head_norm_rope_multi<<<dim3(IDX_HEADS, B), 128, 0, s.stream>>>(
                    q_idx_pf, w.idx_qn.p, s.cos_tab.p, s.sin_tab.p, s.dpos, IDX_HEADS, IDX_DIM, IDX_DIM, IDX_HEADS * IDX_DIM, EPS);

                signed char* l_vkq = s.vk_q + (size_t) ql * KV_VRAM_RING * KVD;
                unsigned short* l_vks = s.vk_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL;
                signed char* l_vvq = s.vv_q + (size_t) ql * KV_VRAM_RING * KVD;
                unsigned short* l_vvs = s.vv_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL;
                signed char* l_hkq = s.hk_q ? (s.hk_q + (size_t) ql * s.ctx_max * KVD) : nullptr;
                unsigned short* l_hks = s.hk_s ? (s.hk_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL) : nullptr;
                signed char* l_hvq = s.hv_q ? (s.hv_q + (size_t) ql * s.ctx_max * KVD) : nullptr;
                unsigned short* l_hvs = s.hv_s ? (s.hv_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL) : nullptr;

                kv_append_int8_multi<<<dim3(8, B), 64, 0, s.stream>>>(
                    kk_pf, vv_pf, l_vkq, l_vks, l_vvq, l_vvs, l_hkq, l_hks, l_hvq, l_hvs, s.dpos, KV_VRAM_RING, s.ctx_max);

                float* l_tail = s.idx_tail.p + (size_t) ql * (IDX_BLK - 1) * IDX_DIM;
                float* l_dead = s.idx_dead.p + (size_t) ql * IDX_DIM;
                float* l_pool = s.idx_pooled.p + (size_t) ql * MAX_IDX_BLOCKS * IDX_DIM;

                qsa_indexer_append_multi<<<1, 128, 0, s.stream>>>(
                    idx_raw_pf, w.idx_kn.p, s.cos_tab.p, s.sin_tab.p, s.dpos, l_tail, l_dead, l_pool, s.ctx_max, EPS, B);
                for (int q0 = 0; q0 < B; q0 += qsa_score_b) {
                    const int qb = std::min(qsa_score_b, B - q0);
                    if (p_start + q0 + qb > IDX_TOPK) {
                        pf_qsa_block_scores_wmma<<<dim3((active_max_b + 63) / 64, (qb + 15) / 16), 256, 0, s.stream>>>(
                            l_pool, q_idx_pf, s.dpos, active_max_b, qsa_sc_pf, q0, qb);
                        pf_qsa_tail_score_kernel<<<qb, 128, 0, s.stream>>>(
                            l_dead, q_idx_pf, s.dpos, active_max_b, qsa_sc_pf, q0);
                    }
                    qsa_block_topk_multi<<<qb, 256, 0, s.stream>>>(
                        qsa_sc_pf, s.dpos, active_max_b, sel_ids_pf, sel_cnt_pf, q0);
                }
                if (B > 2048 && p_start + B > KV_VRAM_RING && l_hkq != nullptr) {
                    // Stage ONLY the middle gap `[KV_SINK_PIN .. p_start + B - (KV_VRAM_RING - KV_SINK_PIN))` into `on_pf`!
                    const int n_mir = (p_start + B) - KV_VRAM_RING;
                    signed char* mir_kq = (signed char*) on_pf;
                    signed char* mir_vq = mir_kq + (((size_t) n_mir * KVD + 255ull) & ~255ull);
                    unsigned short* mir_ks = (unsigned short*) (mir_vq + (((size_t) n_mir * KVD + 255ull) & ~255ull));
                    unsigned short* mir_vs = mir_ks + (((size_t) n_mir * KV_SC_PER_CELL + 127ull) & ~127ull);
                    CHECK(cudaMemcpyAsync(mir_kq, l_hkq + (size_t) KV_SINK_PIN * KVD, (size_t) n_mir * KVD, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(mir_vq, l_hvq + (size_t) KV_SINK_PIN * KVD, (size_t) n_mir * KVD, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(mir_ks, l_hks + (size_t) KV_SINK_PIN * KV_SC_PER_CELL, (size_t) n_mir * KV_SC_PER_CELL * 2, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(mir_vs, l_hvs + (size_t) KV_SINK_PIN * KV_SC_PER_CELL, (size_t) n_mir * KV_SC_PER_CELL * 2, cudaMemcpyHostToDevice, s.stream));
                    pf_qsa_wmma_sparse_attn<<<dim3(KVH, B), 256, 0, s.stream>>>(
                        xn_pf, l_vkq, l_vks, l_vvq, l_vvs,
                        mir_kq - (size_t) KV_SINK_PIN * KVD, mir_ks - (size_t) KV_SINK_PIN * KV_SC_PER_CELL,
                        mir_vq - (size_t) KV_SINK_PIN * KVD, mir_vs - (size_t) KV_SINK_PIN * KV_SC_PER_CELL,
                        sel_ids_pf, s.dpos, xfp16_pf, KV_VRAM_RING);
                } else {
                    pf_qsa_wmma_sparse_attn<<<dim3(KVH, B), 256, 0, s.stream>>>(
                        xn_pf, l_vkq, l_vks, l_vvq, l_vvs, l_hkq, l_hks, l_hvq, l_hvs, sel_ids_pf, s.dpos, xfp16_pf, KV_VRAM_RING);
                }
                dequant_q8p_to_fp16<<<((size_t) NE * QD / 4 + 255) / 256, 256, 0, s.stream>>>(w.wo.p, w.wo.s, s.w_fp16, NE, QD);
                tc_gemm_fp16(NE, QD, B, xfp16_pf, block_pf);
            }
            auto t_l1 = sync_tick();
            if (!w.qsa) ms_gdn += std::chrono::duration<double, std::milli>(t_l1 - t_l0).count();
            else ms_qsa += std::chrono::duration<double, std::milli>(t_l1 - t_l0).count();

            // FFN Half
            gr_write_multi<<<dim3(80, B), 256, 0, s.stream>>>(R_pf, block_pf, inj_pf, NE, HC);
            tc_gr_read(w.hc_norm[1], w.hc_down[1], w.hc_up[1], w.hc_inj[1], nullptr, nullptr, mixed2_pf, m2_fp16_pf, inj_pf);
            bf16_to_fp16_vec<<<((size_t) NEXP * NE / 2 + 255) / 256, 256, 0, s.stream>>>(w.g_inp.p, s.w_fp16, (size_t) NEXP * NE);
            tc_gemm_fp16(NEXP, NE, B, m2_fp16_pf, slog_pf);
            router_top10_fast_multi<<<B, 256, 0, s.stream>>>(slog_pf, sids_pf, swts_pf);

            CHECK(cudaMemsetAsync(exp_cnt_pf, 0, NEXP * 4, s.stream));
            pf_moe_count_kernel<<<(total_k + 255) / 256, 256, 0, s.stream>>>(sids_pf, total_k, exp_cnt_pf);
            pf_moe_scan_meta_kernel<<<1, 512, 0, s.stream>>>(exp_cnt_pf, exp_cur_pf, s.d_lut + l * NEXP, stream_all_ring ? d_meta_pf : d_meta);
            pf_moe_scatter_buckets_kernel<<<(total_k + 255) / 256, 256, 0, s.stream>>>(sids_pf, swts_pf, total_k, exp_cur_pf, btok_pf, bwt_pf);
            if (!stream_all_ring) CHECK(cudaEventRecord(ev_meta, s.stream));

            // Enqueue Shared Expert on `s.stream`
            dequant_q8p_to_fp16<<<((size_t) NFF * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.sh_gate.p, w.sh_gate.s, s.w_fp16, NFF, NE);
            dequant_q8p_to_fp16<<<((size_t) NFF * NE / 4 + 255) / 256, 256, 0, s.stream>>>(w.sh_up.p, w.sh_up.s, s.w_fp16 + (size_t) NFF * NE, NFF, NE);
            tc_gemm_fp16(2 * NFF, NE, B, m2_fp16_pf, sff_pf);
            pf_silu_mul_interleaved_to_fp16<<<(((size_t) B * NFF + 1) / 2 + 255) / 256, 256, 0, s.stream>>>(sff_pf, xfp16_pf, B, NFF);
            dequant_q8p_to_fp16<<<((size_t) NE * NFF / 4 + 255) / 256, 256, 0, s.stream>>>(w.sh_down.p, w.sh_down.s, s.w_fp16, NE, NFF);
            tc_gemm_fp16(NE, NFF, B, xfp16_pf, sshared_pf);
            launch_gemm_bf16(w.s_gate_inp.p, mixed2_pf, sgate_pf, 1, NE, B, s.stream);
            pf_init_shared_comb_kernel<<<((size_t) B * NE + 255) / 256, 256, 0, s.stream>>>(sshared_pf, sgate_pf, comb_pf, B, NE);

            auto t_l2 = sync_tick();
            ms_sh += std::chrono::duration<double, std::milli>(t_l2 - t_l1).count();

            if (stream_all_ring) {
                // ZERO-HOST-SYNC GPU-NATIVE MOE + 512-SLOT CONTINUOUS CROSS-LAYER DMA RING!
                s.total_lookups += total_k;
                const unsigned max_res_tiles = (unsigned) std::min<size_t>(max_tiles_B, (size_t) (total_k / PF_TILE_M + layer_nres[l] + 4));
                if (layer_nres[l] > 0) {
                    pf_build_res_tiles_kernel<<<1, 512, 0, s.stream>>>(d_meta_pf, d_tiles_pf, d_ntiles_pf, d_vhits);
                    pf_moe_gu_wmma_kernel<<<dim3(NFF / 16, max_res_tiles), 256, 0, s.stream>>>(
                        s.xblobs, d_tiles_pf, btok_pf, m2_fp16_pf, ff_fp16_pf, d_ntiles_pf);
                    pf_moe_down_wmma_kernel<<<dim3(NE / 32, max_res_tiles), 256, 0, s.stream>>>(
                        s.xblobs, d_tiles_pf, btok_pf, bwt_pf, ff_fp16_pf, comb_pf, d_ntiles_pf);
                }
                auto t_l3 = (prof_lvl >= 2) ? sync_tick() : t_l2;
                if (prof_lvl >= 2) ms_res += std::chrono::duration<double, std::milli>(t_l3 - t_l2).count();

                for (int wid = layer_w0[l]; wid < layer_w1[l]; ++wid) {
                    const int win = wid & (PF_DMA_WINS - 1);
                    const int w_base = win * PF_WIN_SLOTS;
                    const ColdWavePlan& wp = ring_waves[wid];
                    const unsigned max_cw_tiles = (unsigned) std::min<size_t>(max_tiles_B, (size_t) (total_k / PF_TILE_M + (wp.c1 - wp.c0) + 4));
                    CHECK(cudaStreamWaitEvent(s.stream, ev_dma_w[win], 0));
                    pf_build_cold_wave_tiles_kernel<<<1, 512, 0, s.stream>>>(
                        d_meta_pf, wp.c0, wp.c1, w_base, d_tiles_pf, d_ntiles_pf);
                    pf_moe_gu_wmma_kernel<<<dim3(NFF / 16, max_cw_tiles), 256, 0, s.stream>>>(
                        dma_ring, d_tiles_pf, btok_pf, m2_fp16_pf, ff_fp16_pf, d_ntiles_pf);
                    pf_moe_down_wmma_kernel<<<dim3(NE / 32, max_cw_tiles), 256, 0, s.stream>>>(
                        dma_ring, d_tiles_pf, btok_pf, bwt_pf, ff_fp16_pf, comb_pf, d_ntiles_pf);
                    CHECK(cudaEventRecord(ev_comp_w[win], s.stream));
                    enqueue_ring_wave(wid + PF_DMA_WINS);
                }
                auto t_l4 = sync_tick();
                ms_cold += std::chrono::duration<double, std::milli>(t_l4 - t_l3).count();
            } else {
                // Short-chunk (< 1024 tokens) router-filtered path
                CHECK(cudaEventSynchronize(ev_meta));
                struct ColdExp { int eid; int cnt; int off; };
                std::vector<PfMoeTile> res_tiles;
                std::vector<ColdExp> rem_cold_exps;
                res_tiles.reserve(512);
                rem_cold_exps.reserve(256);
                int layer_hits = 0;
                for (int e = 0; e < NEXP; ++e) {
                    const PfExpMeta m = h_meta[e];
                    if (m.cnt <= 0) continue;
                    if (m.slot >= 0) {
                        layer_hits += m.cnt;
                        for (int sub = 0; sub < m.cnt; sub += PF_TILE_M) {
                            const int wrow = (int) res_tiles.size() * PF_TILE_M;
                            res_tiles.push_back({m.slot, m.off + sub, wrow, std::min(PF_TILE_M, m.cnt - sub)});
                        }
                    } else {
                        rem_cold_exps.push_back({e, m.cnt, m.off});
                    }
                }
                s.total_lookups += total_k;
                s.total_vram_hits += layer_hits;
                const int n_rem = (int) rem_cold_exps.size();
                const int HALF_S = PF_WIN_SLOTS;
                const int n_rem_waves = (n_rem + HALF_S - 1) / HALF_S;
                tot_cold_blobs += n_rem;
                auto enqueue_small = [&](int rw) {
                    if (rw >= n_rem_waves) return;
                    const int win = rw & 1;
                    const int w_base = win * HALF_S;
                    const int c0 = rw * HALF_S, c1 = std::min(n_rem, c0 + HALF_S);
                    CHECK(cudaStreamWaitEvent(dma_stream, ev_comp_win[win], 0));
                    for (int i = c0; i < c1; ++i) {
                        const int gkey = l * NEXP + rem_cold_exps[i].eid;
                        CHECK(cudaMemcpyAsync(dma_ring + (size_t) (w_base + (i - c0)) * PER,
                                              host_pack + (size_t) gkey * PER, PER, cudaMemcpyHostToDevice, dma_stream));
                    }
                    CHECK(cudaEventRecord(ev_dma_win[win], dma_stream));
                };
                enqueue_small(0);
                enqueue_small(1);
                launch_wmma_wave(s.xblobs, res_tiles);
                std::vector<PfMoeTile> wave_tiles;
                for (int rw = 0; rw < n_rem_waves; ++rw) {
                    const int win = rw & 1, w_base = win * HALF_S;
                    const int c0 = rw * HALF_S, c1 = std::min(n_rem, c0 + HALF_S);
                    wave_tiles.clear();
                    for (int i = c0; i < c1; ++i) {
                        const int rslot = w_base + (i - c0), cnt = rem_cold_exps[i].cnt, boff = rem_cold_exps[i].off;
                        for (int sub = 0; sub < cnt; sub += PF_TILE_M) {
                            const int wrow = (int) wave_tiles.size() * PF_TILE_M;
                            wave_tiles.push_back({rslot, boff + sub, wrow, std::min(PF_TILE_M, cnt - sub)});
                        }
                    }
                    CHECK(cudaStreamWaitEvent(s.stream, ev_dma_win[win], 0));
                    launch_wmma_wave(dma_ring, wave_tiles);
                    CHECK(cudaEventRecord(ev_comp_win[win], s.stream));
                    enqueue_small(rw + 2);
                }
            }

            pending_bo = comb_pf;
            pending_inj = inj_pf;
        }
        if (prof_on) {
            std::fprintf(stderr, "[PF-PROF GPU%d B=%d] gdn=%.1fms qsa=%.1fms router+sh=%.1fms moe_res=%.1fms moe_cold=%.1fms (cold_blobs=%lld, %.2f GB)\n",
                         s.dev, B, ms_gdn, ms_qsa, ms_sh, ms_res, ms_cold, tot_cold_blobs, tot_cold_blobs * (double) PER / 1e9);
        }

        if (pending_bo) {
            gr_write_multi<<<dim3(80, B), 256, 0, s.stream>>>(R_pf, pending_bo, pending_inj, NE, HC);
        }
        return R_pf;
    }
};
