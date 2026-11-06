// wzy_qsa_256k.cuh -- 262,144 (256K) Context QSA 4-Head Sparse Indexer (top_k=2048) + INT8 Tiered KV Cache
#pragma once
#include "wzy_kernels_v2.cuh"

static constexpr int KV_GRP = 64;             // INT8 KV group size: 1 fp16 scale per 64 elements
static constexpr int KV_SC_PER_CELL = KVD / KV_GRP; // 512 / 64 = 8 fp16 scales per cell
static constexpr int KV_SINK_PIN = 4096;      // permanently pinned System-Prompt / Attention-Sink cells in GDDR6!
// VRAM-resident KV window. Everything older than this lives in the pinned host mirror and is gathered
// over PCIe by `pf_qsa_wmma_sparse_attn` (12 query heads share one read). Raising it therefore directly
// removes long-context PCIe traffic, at ~34.6 MiB per QSA layer per 32768 cells.
//   build.sh:  -DWZY_KV_RING=65536   ->  +415 MiB/GPU, host gap at 70K drops from 37.9K to 6.7K cells
#ifndef WZY_KV_RING
#define WZY_KV_RING 32768
#endif
static constexpr int KV_VRAM_RING = WZY_KV_RING;    // 4096 pinned sink + sliding ring

__device__ __forceinline__ size_t kv_vram_slot(int p, int ring_cap) {
    if (ring_cap <= KV_SINK_PIN * 2) return (size_t) (p % ring_cap);
    if (p < KV_SINK_PIN) return (size_t) p;
    return (size_t) (KV_SINK_PIN + ((p - KV_SINK_PIN) % (ring_cap - KV_SINK_PIN)));
}

__device__ __forceinline__ bool kv_is_in_vram(int p, int max_written_n_kv, int ring_cap, const void* hk_ptr) {
    if (hk_ptr == nullptr) return true;
    if (ring_cap <= KV_SINK_PIN * 2) return p >= max(0, max_written_n_kv - ring_cap);
    if (p < KV_SINK_PIN) return true;
    const int slide_cap = ring_cap - KV_SINK_PIN;
    return p >= max(KV_SINK_PIN, max_written_n_kv - slide_cap);
}
static constexpr int MAX_CTX_256K = 262144;
static constexpr int MAX_IDX_BLOCKS = MAX_CTX_256K / IDX_BLK + 2; // 65538

// Precompute float64 -> float32 RoPE cos/sin table on host once for [0 .. max_ctx) x 32 pairs
static inline void build_rope_cos_sin_host(int max_ctx, float theta, float* h_cos, float* h_sin) {
    double inv[32];
    for (int i = 0; i < 32; ++i) inv[i] = std::pow((double) theta, -2.0 * (double) i / 64.0);
    for (int p = 0; p < max_ctx; ++p) {
        float* c = h_cos + (size_t) p * 32;
        float* s = h_sin + (size_t) p * 32;
        for (int i = 0; i < 32; ++i) {
            const double ang = (double) p * inv[i];
            c[i] = (float) std::cos(ang);
            s[i] = (float) std::sin(ang);
        }
    }
}

// Fused per-head RMSNorm + Table-based NeoX RoPE (zero FP64 pow/cos/sin on device!)
// Grid: dim3(heads, T), Block: 128
__global__ void fused_head_norm_rope_multi(float* __restrict__ data, const float* __restrict__ gamma,
                                           const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
                                           const int* __restrict__ base_pos_dev,
                                           int heads, int hd, int head_stride, int row_stride, float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int pos = (*base_pos_dev) + t;
    float* p = data + (size_t) t * row_stride + (size_t) h * head_stride;

    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += p[i] * p[i];
    ss = dwsum(ss);
    __shared__ float sh[4];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float tot = (blockDim.x == 128) ? (sh[0] + sh[1] + sh[2] + sh[3]) : sh[0];
        sh[0] = rsqrtf(tot / (float) hd + eps);
    }
    __syncthreads();
    const float sc = sh[0];
    for (int i = threadIdx.x; i < hd; i += blockDim.x) p[i] *= sc * gamma[i];
    __syncthreads();

    if (threadIdx.x < 32) {
        const int i = threadIdx.x;
        const float cs = __ldg(cos_tab + (size_t) pos * 32 + i);
        const float sn = __ldg(sin_tab + (size_t) pos * 32 + i);
        const float a = p[i], d = p[i + 32];
        p[i] = a * cs - d * sn;
        p[i + 32] = a * sn + d * cs;
    }
}

// Quantize & append T cells of K [T, 512] and V [T, 512] into INT8 VRAM ring + optional UVA Host full mirror
// Grid: dim3(8, T) (8 groups of 64 elements = 512), Block: 64
__global__ void kv_append_int8_multi(const float* __restrict__ k_in, const float* __restrict__ v_in,
                                     signed char* __restrict__ vk_q, unsigned short* __restrict__ vk_s,
                                     signed char* __restrict__ vv_q, unsigned short* __restrict__ vv_s,
                                     signed char* __restrict__ hk_q, unsigned short* __restrict__ hk_s,
                                     signed char* __restrict__ hv_q, unsigned short* __restrict__ hv_s,
                                     const int* __restrict__ base_pos_dev, int ring_cap, int max_ctx) {
    const int grp = blockIdx.x; // 0..7 (each group is 64 floats)
    const int t = blockIdx.y;
    const int lane = threadIdx.x; // 0..63
    const int pos = (*base_pos_dev) + t;
    if (pos < 0 || pos >= max_ctx) return;

    const int idx = grp * 64 + lane;
    const float kv = k_in[(size_t) t * KVD + idx];
    const float vv = v_in[(size_t) t * KVD + idx];

    // Reduce max abs across 64 threads (2 warps)
    __shared__ float smx[4];
    float km = fabsf(kv), vm = fabsf(vv);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        km = fmaxf(km, __shfl_xor_sync(0xffffffffu, km, o));
        vm = fmaxf(vm, __shfl_xor_sync(0xffffffffu, vm, o));
    }
    if ((lane & 31) == 0) {
        smx[lane >> 5] = km;
        smx[2 + (lane >> 5)] = vm;
    }
    __syncthreads();
    const float kmax = fmaxf(smx[0], smx[1]);
    const float vmax = fmaxf(smx[2], smx[3]);
    const unsigned short ksh = df2h(kmax / 127.0f);
    const unsigned short vsh = df2h(vmax / 127.0f);
    const float kd = dh2f(ksh), vd = dh2f(vsh);

    const int kqi = kd > 0.f ? max(-127, min(127, __float2int_rn(kv / kd))) : 0;
    const int vqi = vd > 0.f ? max(-127, min(127, __float2int_rn(vv / vd))) : 0;

    if (kv_is_in_vram(pos, (*base_pos_dev) + (int) gridDim.y, ring_cap, (const void*) 1)) {
        const size_t rslot = kv_vram_slot(pos, ring_cap);
        vk_q[rslot * KVD + idx] = (signed char) kqi;
        vv_q[rslot * KVD + idx] = (signed char) vqi;
        if (lane == 0) {
            vk_s[rslot * KV_SC_PER_CELL + grp] = ksh;
            vv_s[rslot * KV_SC_PER_CELL + grp] = vsh;
        }
    }

    if (hk_q != nullptr) {
        const size_t hslot = (size_t) pos;
        hk_q[hslot * KVD + idx] = (signed char) kqi;
        hv_q[hslot * KVD + idx] = (signed char) vqi;
        if (lane == 0) {
            hk_s[hslot * KV_SC_PER_CELL + grp] = ksh;
            hv_s[hslot * KV_SC_PER_CELL + grp] = vsh;
        }
    }
}

// QSA Sparse Indexer key append (exact match to native_qsa_indexer.cu::append_body, supports T >= 1 steps)
// Grid: 1, Block: 128
__global__ void qsa_indexer_append_multi(const float* __restrict__ idx_raw,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
                                         const int* __restrict__ base_pos_dev,
                                         float* __restrict__ tail, float* __restrict__ dead,
                                         float* __restrict__ pooled, int max_ctx, float eps, int T,
                                         float* __restrict__ tail_snaps = nullptr,
                                         float* __restrict__ dead_snaps = nullptr,
                                         float* __restrict__ pool_snaps = nullptr) {
    __shared__ float values[IDX_DIM];
    __shared__ float partials[4];
    const int d = threadIdx.x; // 0..127
    const int base_pos = *base_pos_dev;

    for (int step = 0; step < T; ++step) {
        const int pos = base_pos + step;
        if (pos >= 0 && pos < max_ctx) {
            const float incoming = __half2float(__float2half_rn(idx_raw[(size_t) step * IDX_DIM + d]));
            const int slot = pos % IDX_BLK;
            if (slot < IDX_BLK - 1) tail[slot * IDX_DIM + d] = incoming;

            if (pos == 0 || slot == IDX_BLK - 1) {
                float sum = (pos == 0) ? incoming : tail[d];
                #pragma unroll
                for (int j = 1; j < IDX_BLK; ++j)
                    sum = __fadd_rn(sum, (pos == 0 || j == IDX_BLK - 1) ? incoming : tail[j * IDX_DIM + d]);
                const float mean = __fmaf_rn(0.25f, sum, 0.0f);

                float sq = dwsum(mean * mean);
                if ((d & 31) == 0) partials[d >> 5] = sq;
                __syncthreads();
                const float ss = partials[0] + partials[1] + partials[2] + partials[3];
                const float sc = rsqrtf(ss / (float) IDX_DIM + eps);
                values[d] = sc * mean * gamma[d];
                __syncthreads();

                const int b = pos / IDX_BLK;
                const int rope_pos = (pos == 0) ? 0 : (IDX_BLK * b);
                float y = values[d];
                if (d < NROT) {
                    const int pair = d & 31;
                    const float c = __ldg(cos_tab + (size_t) rope_pos * 32 + pair);
                    const float s = __ldg(sin_tab + (size_t) rope_pos * 32 + pair);
                    const float a = values[pair], z = values[pair + 32];
                    y = (d < 32) ? (a * c - z * s) : (a * s + z * c);
                }
                pooled[(size_t) b * IDX_DIM + d] = y;
                if (pos == 0) dead[d] = y;
                else pooled[(size_t) (b + 1) * IDX_DIM + d] = dead[d];
            }
            if (tail_snaps != nullptr) {
                #pragma unroll
                for (int j = 0; j < IDX_BLK - 1; ++j)
                    tail_snaps[((size_t) step * (IDX_BLK - 1) + j) * IDX_DIM + d] = tail[j * IDX_DIM + d];
                dead_snaps[(size_t) step * IDX_DIM + d] = dead[d];
                const int b = pos / IDX_BLK;
                pool_snaps[((size_t) step * 2 + 0) * IDX_DIM + d] = pooled[(size_t) b * IDX_DIM + d];
                pool_snaps[((size_t) step * 2 + 1) * IDX_DIM + d] = pooled[(size_t) (b + 1) * IDX_DIM + d];
            }
        }
        if (T > 1) __syncthreads();
    }
}

__global__ void qsa_restore_snap_kernel(float* __restrict__ idx_tail, float* __restrict__ idx_dead, float* __restrict__ idx_pooled,
                                        const float* __restrict__ tail_snaps, const float* __restrict__ dead_snaps, const float* __restrict__ pool_snaps,
                                        int acc_idx, int acc_pos) {
    const int ql = blockIdx.x;
    const int d = threadIdx.x; // 0..127
    const size_t snap_slot = (size_t) ql * MAX_W + acc_idx;
    #pragma unroll
    for (int j = 0; j < IDX_BLK - 1; ++j)
        idx_tail[((size_t) ql * (IDX_BLK - 1) + j) * IDX_DIM + d] = tail_snaps[(snap_slot * (IDX_BLK - 1) + j) * IDX_DIM + d];
    idx_dead[(size_t) ql * IDX_DIM + d] = dead_snaps[snap_slot * IDX_DIM + d];
    const int b = max(0, acc_pos) / IDX_BLK;
    idx_pooled[((size_t) ql * MAX_IDX_BLOCKS + b) * IDX_DIM + d] = pool_snaps[(snap_slot * 2 + 0) * IDX_DIM + d];
    idx_pooled[((size_t) ql * MAX_IDX_BLOCKS + b + 1) * IDX_DIM + d] = pool_snaps[(snap_slot * 2 + 1) * IDX_DIM + d];
}

__device__ __forceinline__ uint32_t qsa_order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Score all pooled 4-token blocks against 4-head q_idx (exact match to qsa_select.cu::block_scores_kernel)
// Grid: dim3((max_active_blocks + 7) / 8, T), Block: 256 (8 warps)
__global__ void qsa_block_scores_multi(const float* __restrict__ pooled, const float* __restrict__ dead,
                                       const float* __restrict__ q_idx, const int* __restrict__ base_pos_dev,
                                       int max_blocks, float* __restrict__ scores_out, int qi_off = 0) {
    const int qi = qi_off + blockIdx.y;
    const int n_kv = (*base_pos_dev) + qi + 1;
    if (n_kv <= IDX_TOPK) return; // short context skips scoring entirely!

    const int n_bid = n_kv / IDX_BLK;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float* q = q_idx + (size_t) qi * (IDX_HEADS * IDX_DIM) + lane * 4;
    float4 q4[IDX_HEADS];
    #pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) q4[h] = *(const float4*) (q + h * IDX_DIM);

    for (int b = blockIdx.x * 8 + warp; b <= n_bid && b < max_blocks; b += gridDim.x * 8) {
        const float* key = (b == n_bid) ? dead : (pooled + (size_t) b * IDX_DIM);
        const float4 k4 = *(const float4*) (key + lane * 4);
        float score = 0.0f;
        #pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) {
            float d = dwsum(k4.x * q4[h].x + k4.y * q4[h].y + k4.z * q4[h].z + k4.w * q4[h].w);
            score += d > 0.0f ? d : 0.0f;
        }
        if (lane == 0) {
            if (b == n_bid && (n_kv % IDX_BLK) != 0) score += 1e9f;
            scores_out[(size_t) blockIdx.y * max_blocks + b] = score;
        }
    }
}

// 4-pass 8-bit radix top-2048 cell selector (exact match to qsa_select.cu::block_topk_kernel)
// Grid: T, Block: 256
__global__ void qsa_block_topk_multi(const float* __restrict__ scores, const int* __restrict__ base_pos_dev,
                                     int max_blocks, int* __restrict__ sel_ids, int* __restrict__ sel_cnt, int qi_off = 0) {
    __shared__ int hist[256];
    __shared__ int s_digit, s_above;

    const int qi = qi_off + blockIdx.x;
    const int n_kv = (*base_pos_dev) + qi + 1;
    const int width = min(n_kv, IDX_TOPK);
    int* out = sel_ids + (size_t) qi * IDX_TOPK;
    const int t = threadIdx.x;

    if (t == 0) sel_cnt[qi] = width;
    if (n_kv <= IDX_TOPK) {
        for (int j = t; j < n_kv; j += 256) out[j] = j;
        return;
    }

    const int n_bid = n_kv / IDX_BLK;
    const float* sc = scores + (size_t) blockIdx.x * max_blocks;
    const int nb = n_bid + 1;
    auto weight = [&](int b) -> int { return b < n_bid ? IDX_BLK : (n_kv - n_bid * IDX_BLK); };

    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        const uint32_t hi_mask = (shift == 24) ? 0u : (0xffffffffu << (shift + 8));
        int prev_d = -1, prev_w = 0;
        for (int b = t; b < nb; b += 256) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = qsa_order_key(__ldg(sc + b));
            if ((k & hi_mask) == (prefix & hi_mask)) {
                const int d = (int) ((k >> shift) & 255u);
                if (d == prev_d) {
                    prev_w += w;
                } else {
                    if (prev_d >= 0) atomicAdd(&hist[prev_d], prev_w);
                    prev_d = d;
                    prev_w = w;
                }
            }
        }
        if (prev_d >= 0) atomicAdd(&hist[prev_d], prev_w);
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }

    const uint32_t thr = prefix;
    const int eq_budget = width - above;
    int gt = 0, eq = 0;
    for (int b = t; b < nb; b += 256) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = qsa_order_key(__ldg(sc + b));
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }

    const int lane = t & 31, warp = t >> 5;
    __shared__ int wtot_g[8], wtot_e[8];
    int sg = gt, se = eq;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ng = __shfl_up_sync(0xffffffffu, sg, o);
        const int ne = __shfl_up_sync(0xffffffffu, se, o);
        if (lane >= o) { sg += ng; se += ne; }
    }
    if (lane == 31) { wtot_g[warp] = sg; wtot_e[warp] = se; }
    __syncthreads();
    if (warp == 0) {
        int wg = (lane < 8) ? wtot_g[lane] : 0;
        int we = (lane < 8) ? wtot_e[lane] : 0;
        #pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            const int ng = __shfl_up_sync(0xffffffffu, wg, o);
            const int ne = __shfl_up_sync(0xffffffffu, we, o);
            if (lane >= o) { wg += ng; we += ne; }
        }
        if (lane < 8) {
            wtot_g[lane] = wg - wtot_g[lane];
            wtot_e[lane] = we - wtot_e[lane];
        }
    }
    __syncthreads();
    const int excl_e = wtot_e[warp] + (se - eq);
    const int my_eq = max(0, min(eq, eq_budget - excl_e));
    const int sel = gt + my_eq;

    int s_sel = sel;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ns = __shfl_up_sync(0xffffffffu, s_sel, o);
        if (lane >= o) s_sel += ns;
    }
    if (lane == 31) wtot_g[warp] = s_sel;
    __syncthreads();
    if (warp == 0) {
        int ws = (lane < 8) ? wtot_g[lane] : 0;
        #pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            const int ns = __shfl_up_sync(0xffffffffu, ws, o);
            if (lane >= o) ws += ns;
        }
        if (lane < 8) wtot_g[lane] = ws - wtot_g[lane];
    }
    __syncthreads();

    int wpos = wtot_g[warp] + (s_sel - sel);
    int eq_left = my_eq;
    for (int b = t; b < nb; b += 256) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = qsa_order_key(__ldg(sc + b));
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = b * IDX_BLK + c;
        } else if (k == thr && eq_left > 0) {
            const int take = min(w, eq_left);
            for (int c = 0; c < take; ++c) out[wpos++] = b * IDX_BLK + c;
            eq_left -= take;
        }
    }
}

// Sparse INT8 Tiered Decode Attention (attends over `width <= 2048` selected cells, parallel softmax & V-weighted sum!)
// Grid: dim3(QH, T), Block: 256
__global__ void qsa_attend_int8_sparse_multi(
    const float* __restrict__ qg,
    const signed char* __restrict__ vk_q, const unsigned short* __restrict__ vk_s,
    const signed char* __restrict__ vv_q, const unsigned short* __restrict__ vv_s,
    const signed char* __restrict__ hk_q, const unsigned short* __restrict__ hk_s,
    const signed char* __restrict__ hv_q, const unsigned short* __restrict__ hv_s,
    const int* __restrict__ sel_ids, const int* __restrict__ base_pos_dev,
    float* __restrict__ out, int ring_cap) {
    __shared__ float sc[IDX_TOPK]; // 2048 * 4 = 8 KB shared memory!
    __shared__ float qsh[HD];      // 256 * 4 = 1 KB shared memory!
    __shared__ float wr[8];
    const int h = blockIdx.x;  // 0..23
    const int qi = blockIdx.y; // 0..T-1
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int kv_head = h / (QH / KVH); // 0 or 1
    const int head_off = kv_head * HD;
    const int sc_off = kv_head * (HD / KV_GRP); // 0 or 4

    const int cur_pos = (*base_pos_dev) + qi;
    const int n_kv = cur_pos + 1;
    const int width = min(n_kv, IDX_TOPK);
    const int max_written_n_kv = max(n_kv, (*base_pos_dev) + (int) gridDim.y);

    const float* qrow = qg + ((size_t) qi * QH + h) * (2 * HD);
    qsh[tid] = qrow[tid];
    __syncthreads();

    const float inv_sqrt_hd = 0.0625f; // 1 / sqrt(256)
    const int* ids = sel_ids + (size_t) qi * IDX_TOPK;

    // 1. Coalesced 8-warp parallel scoring: each warp processes 1 cell `idx` per step, reading 256 contiguous bytes!
    const int d0 = lane * 8;
    const int g_lane = lane >> 3;
    const float4 qf0 = *(const float4*) (qsh + d0);
    const float4 qf1 = *(const float4*) (qsh + d0 + 4);
    float local_max = -1e30f;
    for (int idx = warp; idx < width; idx += 8) {
        const int p = (n_kv <= IDX_TOPK) ? idx : ids[idx];
        const bool in_vram = kv_is_in_vram(p, max_written_n_kv, ring_cap, hk_q);
        const size_t vslot = kv_vram_slot(p, ring_cap);
        const signed char* kp = in_vram
            ? (vk_q + vslot * KVD + head_off)
            : (hk_q + (size_t) p * KVD + head_off);
        const unsigned short* ks = in_vram
            ? (vk_s + vslot * KV_SC_PER_CELL + sc_off)
            : (hk_s + (size_t) p * KV_SC_PER_CELL + sc_off);

        const char4 kv0 = *(const char4*) (kp + d0);
        const char4 kv1 = *(const char4*) (kp + d0 + 4);
        const float part = dh2f(ks[g_lane]) * (
              (float) kv0.x * qf0.x + (float) kv0.y * qf0.y + (float) kv0.z * qf0.z + (float) kv0.w * qf0.w
            + (float) kv1.x * qf1.x + (float) kv1.y * qf1.y + (float) kv1.z * qf1.z + (float) kv1.w * qf1.w);
        const float s = dwsum(part) * inv_sqrt_hd;
        if (lane == 0) {
            sc[idx] = s;
            local_max = fmaxf(local_max, s);
        }
    }

    // 2. Parallel block-wide max + softmax denominator
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
    if (lane == 0) wr[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? wr[lane] : -1e30f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) wr[0] = m;
    }
    __syncthreads();
    const float mx = wr[0];

    float local_sum = 0.f;
    for (int idx = tid; idx < width; idx += 256) {
        const float e = __expf(sc[idx] - mx);
        sc[idx] = e;
        local_sum += e;
    }
    local_sum = dwsum(local_sum);
    if (lane == 0) wr[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < 8) ? wr[lane] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) wr[0] = 1.0f / s;
    }
    __syncthreads();
    const float inv_sum = wr[0];

    // 3. Coalesced 8-warp parallel V accumulation: each warp processes `idx = warp, warp + 8, ...` (8x faster!)
    float vacc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int idx = warp; idx < width; idx += 8) {
        const int p = (n_kv <= IDX_TOPK) ? idx : ids[idx];
        const bool in_vram = kv_is_in_vram(p, max_written_n_kv, ring_cap, hv_q);
        const size_t vslot = kv_vram_slot(p, ring_cap);
        const signed char* vp = in_vram
            ? (vv_q + vslot * KVD + head_off)
            : (hv_q + (size_t) p * KVD + head_off);
        const unsigned short* vs = in_vram
            ? (vv_s + vslot * KV_SC_PER_CELL + sc_off)
            : (hv_s + (size_t) p * KV_SC_PER_CELL + sc_off);
        const float w_sc = sc[idx] * dh2f(vs[g_lane]);
        const char4 vv0 = *(const char4*) (vp + d0);
        const char4 vv1 = *(const char4*) (vp + d0 + 4);
        vacc[0] = fmaf(w_sc, (float) vv0.x, vacc[0]);
        vacc[1] = fmaf(w_sc, (float) vv0.y, vacc[1]);
        vacc[2] = fmaf(w_sc, (float) vv0.z, vacc[2]);
        vacc[3] = fmaf(w_sc, (float) vv0.w, vacc[3]);
        vacc[4] = fmaf(w_sc, (float) vv1.x, vacc[4]);
        vacc[5] = fmaf(w_sc, (float) vv1.y, vacc[5]);
        vacc[6] = fmaf(w_sc, (float) vv1.z, vacc[6]);
        vacc[7] = fmaf(w_sc, (float) vv1.w, vacc[7]);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 8; ++j) sc[warp * 256 + d0 + j] = vacc[j];
    __syncthreads();
    float acc = sc[0 * 256 + tid] + sc[1 * 256 + tid] + sc[2 * 256 + tid] + sc[3 * 256 + tid]
              + sc[4 * 256 + tid] + sc[5 * 256 + tid] + sc[6 * 256 + tid] + sc[7 * 256 + tid];
    acc *= inv_sum;
    const float gate = qrow[HD + tid];
    out[((size_t) qi * QH + h) * HD + tid] = acc / (1.0f + __expf(-gate));
}

// GQA-12 Online Flash-Softmax Sparse INT8 Attention (processes all 12 query heads sharing 1 KV head in ONE block!)
// Cuts GDDR6 and 256K PCIe KV traffic by 12.0x!
// Grid: dim3(KVH=2, T), Block: 256
__global__ void __launch_bounds__(256) qsa_attend_int8_gqa12_pf_kernel(
    const float* __restrict__ qg,
    const signed char* __restrict__ vk_q, const unsigned short* __restrict__ vk_s,
    const signed char* __restrict__ vv_q, const unsigned short* __restrict__ vv_s,
    const signed char* __restrict__ hk_q, const unsigned short* __restrict__ hk_s,
    const signed char* __restrict__ hv_q, const unsigned short* __restrict__ hv_s,
    const int* __restrict__ sel_ids, const int* __restrict__ base_pos_dev,
    float* __restrict__ out, int ring_cap) {

    __shared__ float qsh[12][HD];
    __shared__ float s_scores[8][12];
    __shared__ float s_alpha[12];
    __shared__ float s_w[8][12];

    const int kv_head = blockIdx.x; // 0 or 1
    const int qi = blockIdx.y;      // 0 .. T-1
    const int tid = threadIdx.x;    // 0 .. 255
    const int lane = tid & 31, warp = tid >> 5;
    const int h_base = kv_head * 12;
    const int head_off = kv_head * HD;
    const int sc_off = kv_head * (HD / KV_GRP);

    const int cur_pos = (*base_pos_dev) + qi;
    const int n_kv = cur_pos + 1;
    const int width = min(n_kv, IDX_TOPK);
    const int ring_floor = max(0, n_kv - ring_cap);
    const int* ids = sel_ids + (size_t) qi * IDX_TOPK;

    #pragma unroll
    for (int h = 0; h < 12; ++h) {
        qsh[h][tid] = qg[((size_t) qi * QH + (h_base + h)) * (2 * HD) + tid] * 0.0625f;
    }
    __syncthreads();

    float acc_o[12];
    #pragma unroll
    for (int h = 0; h < 12; ++h) acc_o[h] = 0.f;

    float m_i = -1e30f, l_i = 0.f;
    const int d0 = lane * 8;
    const int g_lane = lane >> 3;
    const int vg = tid >> 6;

    for (int base_idx = 0; base_idx < width; base_idx += 8) {
        const int step_cells = min(8, width - base_idx);

        if (warp < step_cells) {
            const int idx = base_idx + warp;
            const int p = (n_kv <= IDX_TOPK) ? idx : ids[idx];
            const bool in_vram = (p >= ring_floor) || (hk_q == nullptr);
            const signed char* kp = in_vram
                ? (vk_q + (size_t) (p % ring_cap) * KVD + head_off)
                : (hk_q + (size_t) p * KVD + head_off);
            const unsigned short* ks = in_vram
                ? (vk_s + (size_t) (p % ring_cap) * KV_SC_PER_CELL + sc_off)
                : (hk_s + (size_t) p * KV_SC_PER_CELL + sc_off);

            const char4 kv0 = *(const char4*) (kp + d0);
            const char4 kv1 = *(const char4*) (kp + d0 + 4);
            const float ksc = dh2f(ks[g_lane]);
            const float kf0 = ksc * (float) kv0.x, kf1 = ksc * (float) kv0.y, kf2 = ksc * (float) kv0.z, kf3 = ksc * (float) kv0.w;
            const float kf4 = ksc * (float) kv1.x, kf5 = ksc * (float) kv1.y, kf6 = ksc * (float) kv1.z, kf7 = ksc * (float) kv1.w;

            #pragma unroll
            for (int h = 0; h < 12; ++h) {
                const float4 qf0 = *(const float4*) (&qsh[h][d0]);
                const float4 qf1 = *(const float4*) (&qsh[h][d0 + 4]);
                const float part = kf0 * qf0.x + kf1 * qf0.y + kf2 * qf0.z + kf3 * qf0.w
                                 + kf4 * qf1.x + kf5 * qf1.y + kf6 * qf1.z + kf7 * qf1.w;
                const float dot = dwsum(part);
                if (lane == 0) s_scores[warp][h] = dot;
            }
        }
        __syncthreads();

        if (tid < 12) {
            const int h = tid;
            float m_new = m_i;
            for (int c = 0; c < step_cells; ++c) m_new = fmaxf(m_new, s_scores[c][h]);
            const float alpha = __expf(m_i - m_new);
            s_alpha[h] = alpha;
            float l_new = l_i * alpha;
            for (int c = 0; c < step_cells; ++c) {
                const float w = __expf(s_scores[c][h] - m_new);
                s_w[c][h] = w;
                l_new += w;
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < 12; ++h) acc_o[h] *= s_alpha[h];

        for (int c = 0; c < step_cells; ++c) {
            const int idx = base_idx + c;
            const int p = (n_kv <= IDX_TOPK) ? idx : ids[idx];
            const bool in_vram = (p >= ring_floor) || (hv_q == nullptr);
            const signed char vcode = in_vram
                ? vv_q[(size_t) (p % ring_cap) * KVD + head_off + tid]
                : hv_q[(size_t) p * KVD + head_off + tid];
            const unsigned short vsh = in_vram
                ? vv_s[(size_t) (p % ring_cap) * KV_SC_PER_CELL + sc_off + vg]
                : hv_s[(size_t) p * KV_SC_PER_CELL + sc_off + vg];
            const float v_val = dh2f(vsh) * (float) vcode;
            #pragma unroll
            for (int h = 0; h < 12; ++h) acc_o[h] = fmaf(s_w[c][h], v_val, acc_o[h]);
        }
        __syncthreads();
    }

    if (tid < 12) s_alpha[tid] = 1.0f / fmaxf(l_i, 1e-20f);
    __syncthreads();

    #pragma unroll
    for (int h = 0; h < 12; ++h) {
        const float gate = qg[((size_t) qi * QH + (h_base + h)) * (2 * HD) + HD + tid];
        out[((size_t) qi * QH + (h_base + h)) * HD + tid] = (acc_o[h] * s_alpha[h]) / (1.0f + __expf(-gate));
    }
}
