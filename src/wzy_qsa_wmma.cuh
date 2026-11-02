#pragma once
#include "wzy_qsa_256k.cuh"
#include <mma.h>

__global__ void pf_gdn_conv_parallel(const float* __restrict__ prev_hist,
                                     const float* __restrict__ qkv,
                                     const float* __restrict__ kw,
                                     float* __restrict__ out_h,
                                     int width, int qk_groups, float eps, int num_steps) {
    __shared__ float red[4];
    const int grp = blockIdx.x;
    const int step = blockIdx.y;
    const int ch = grp * 128 + threadIdx.x;

    const float v0 = (step >= 3) ? qkv[(size_t) (step - 3) * width + ch] : prev_hist[ch * 3 + step];
    const float v1 = (step >= 2) ? qkv[(size_t) (step - 2) * width + ch] : ((step == 1) ? prev_hist[ch * 3 + 2] : prev_hist[ch * 3 + 1]);
    const float v2 = (step >= 1) ? qkv[(size_t) (step - 1) * width + ch] : prev_hist[ch * 3 + 2];
    const float vx = qkv[(size_t) step * width + ch];

    const float csum = v0 * kw[ch * 4 + 0] + v1 * kw[ch * 4 + 1] + v2 * kw[ch * 4 + 2] + vx * kw[ch * 4 + 3];
    float y = csum / (1.0f + __expf(-csum));
    if (grp < qk_groups) {
        float sq = dwsum(y * y);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float tot = red[0] + red[1] + red[2] + red[3];
        y *= rsqrtf(tot + eps);
    }
    out_h[(size_t) step * width + ch] = y;
}

__global__ void pf_gdn_conv_update_state(float* __restrict__ hist, const float* __restrict__ qkv,
                                         int width, int num_steps) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= width) return;
    const float h1 = hist[ch * 3 + 1], h2 = hist[ch * 3 + 2];
    float n0, n1, n2;
    if (num_steps >= 3) {
        n0 = qkv[(size_t) (num_steps - 3) * width + ch];
        n1 = qkv[(size_t) (num_steps - 2) * width + ch];
        n2 = qkv[(size_t) (num_steps - 1) * width + ch];
    } else if (num_steps == 2) {
        n0 = h2;
        n1 = qkv[0 * width + ch];
        n2 = qkv[1 * width + ch];
    } else {
        n0 = h1;
        n1 = h2;
        n2 = qkv[0 * width + ch];
    }
    hist[ch * 3 + 0] = n0;
    hist[ch * 3 + 1] = n1;
    hist[ch * 3 + 2] = n2;
}

// Zero-Shuffle 3-Value-Heads-per-Key-Head GDN Recurrence:
// Grid: 64 blocks (16 key heads x 4 column blocks of 32 columns -> fits in ONE SINGLE WAVE on 68 SMs!)
// Block: dim3(32, 4) = 128 threads (`c = threadIdx.x` column 0..31, `rg = threadIdx.y` row-group 0..3 of 32 rows)
// Within each warp (`rg`), all 32 lanes read the EXACT SAME `sk` and `sq` element (1-cycle shared-memory broadcast!)
// and hold all 3 value heads (`vh = qh + j * 16`, `j = 0..2`) in registers (`s[3][32]`), with ZERO warp shuffles!
__global__ void __launch_bounds__(128, 2) pf_gdn_recur_kh_kernel(
    float* __restrict__ state, const float* __restrict__ h_qkv,
    const float* __restrict__ decay, const float* __restrict__ bt,
    float* __restrict__ raw_out, int num_steps) {

    constexpr int S = 128, HK = 16, HV = 48, C = 10240;
    constexpr int RG = 4, RPG = 32, CB = 32, NCB = 4, VPK = 3, TB = 8;
    constexpr int NT = CB * RG, QKP = S / 4, VP = CB / 4;

    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][CB];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK];
    __shared__ float rkv[VPK][RG][CB], ro[VPK][RG][CB];

    const int qh = blockIdx.x / NCB;
    const int cb = blockIdx.x % NCB;
    const int c = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * CB + c;
    const int col = cb * CB + c;
    const size_t rs = (size_t) HV * S;

    float s[VPK][RPG];
    #pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float* base = state + ((size_t) (rg * RPG) * HV + (qh + j * HK)) * S + col;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) s[j][r] = base[r * rs];
    }

    const int nblk = (num_steps + TB - 1) / TB;
    auto stage = [&](int k) {
        const int bb = k & 1;
        const int t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < num_steps) {
                const float* src = h_qkv + (size_t) (t0 + i) * C + (isk ? HK * S : 0) + qh * S + jj;
                *(float4*) (isk ? &sk[bb][i][jj] : &sq[bb][i][jj]) = *(const float4*) src;
            }
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < num_steps) {
                const float* src = h_qkv + (size_t) (t0 + i) * C + 2 * HK * S + (qh + j * HK) * S + cb * CB + jj;
                *(float4*) (&sv[bb][i][j][jj]) = *(const float4*) src;
            }
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < num_steps) {
                const float val = (isb ? bt : decay)[(size_t) (t0 + i) * HV + (qh + j * HK)];
                if (isb) sb[bb][i][j] = val;
                else sg[bb][i][j] = val;
            }
        }
    };

    if (nblk > 0) stage(0);
    for (int k = 0; k < nblk; ++k) {
        __syncthreads();
        if (k + 1 < nblk) stage(k + 1);
        const int bb = k & 1;
        const int n = min(TB, num_steps - k * TB);
        for (int i = 0; i < n; ++i) {
            const int t = k * TB + i;
            float kc[RPG];
            #pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
            #pragma unroll
            for (int j = 0; j < VPK; ++j) { g[j] = sg[bb][i][j]; kv[j] = 0.0f; o[j] = 0.0f; }
            #pragma unroll
            for (int r = 0; r < RPG; ++r) {
                #pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] *= g[j];
                    kv[j] = fmaf(s[j][r], kc[r], kv[j]);
                }
            }
            #pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float kv_col = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - kv_col) * sb[bb][i][j];
            }
            #pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
                #pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = fmaf(kc[r], delta[j], s[j][r]);
                    o[j] = fmaf(s[j][r], qr, o[j]);
                }
            }
            #pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            __syncthreads();
            if (rg < VPK) {
                raw_out[(size_t) t * (HV * S) + (size_t) (qh + rg * HK) * S + col] =
                    (ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c]) * 0.08838834764831845f;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float* base = state + ((size_t) (rg * RPG) * HV + (qh + j * HK)) * S + col;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs] = s[j][r];
    }
}

// Parallel GDN output RMSNorm + SiLU gate across all `(head, step)` simultaneously!
// Grid: dim3(VH=48, num_steps), Block: 128
__global__ void pf_gdn_rmsnorm_gate_parallel(const float* __restrict__ raw_out,
                                             const float* __restrict__ z,
                                             const float* __restrict__ gamma,
                                             __half* __restrict__ y_fp16_out, int hv, float eps) {
    __shared__ float wsum[4];
    const int head = blockIdx.x;
    const int step = blockIdx.y;
    const int col = threadIdx.x; // 0..127
    const size_t idx = ((size_t) step * hv + head) * 128 + col;
    const float oc = raw_out[idx];
    const float sq = dwsum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sq;
    __syncthreads();
    const float tot = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float sc = rsqrtf(tot * (1.0f / 128.0f) + eps);
    const float zv = z[idx];
    y_fp16_out[idx] = __float2half(oc * sc * gamma[col] / (1.0f + __expf(-zv)));
}

__global__ void pf_gr_norm_fp16(float* __restrict__ res_state, const float* __restrict__ w_norm,
                                const float* __restrict__ prev_bo, const float* __restrict__ prev_inj,
                                __half* __restrict__ xn_out, int n_embd, int hc, float eps) {
    const int stream_id = blockIdx.x;
    const int step = blockIdx.y;
    const size_t base = ((size_t) step * hc + stream_id) * n_embd;
    float* rc = res_state + base;
    const float* gc = w_norm + (size_t) stream_id * n_embd;
    __half* dst = xn_out + base;
    const float gate_val = prev_bo ? (2.0f / (1.0f + __expf(-prev_inj[step * hc + stream_id] / (float) hc))) : 0.0f;
    const float* bop = prev_bo ? (prev_bo + (size_t) step * n_embd) : nullptr;

    float sq = 0.f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float val = rc[d];
        if (bop) { val += bop[d] * gate_val; rc[d] = val; }
        sq += val * val;
    }
    __shared__ float part[8];
    sq = dwsum(sq);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x < 8) sq = part[threadIdx.x]; else sq = 0.f;
    if ((threadIdx.x >> 5) == 0) {
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (threadIdx.x == 0) part[0] = rsqrtf(sq / (float) n_embd + eps);
    }
    __syncthreads();
    const float inv_rms = part[0];
    for (int d2 = threadIdx.x * 2; d2 < n_embd; d2 += blockDim.x * 2) {
        const float y0 = rc[d2] * inv_rms * gc[d2];
        const float y1 = rc[d2 + 1] * inv_rms * gc[d2 + 1];
        *(__half2*) (dst + d2) = __floats2half2_rn(y0, y1);
    }
}

__global__ void pf_gr_up_mean_fp16(const __half* __restrict__ xn_in, const __half* __restrict__ gg_in,
                                   float* __restrict__ mixed_out, __half* __restrict__ mixed_fp16_out,
                                   int num_steps, int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_steps * n_embd) return;
    const int step = idx / n_embd;
    const int d = idx - step * n_embd;
    const size_t base = (size_t) step * 4 * n_embd + d;
    float acc = 0.f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float xv = __half2float(xn_in[base + (size_t) c * n_embd]);
        const float gv = __half2float(gg_in[base + (size_t) c * n_embd]);
        acc += xv / (1.0f + __expf(-gv));
    }
    const float val = acc * 0.25f;
    if (mixed_out) mixed_out[idx] = val;
    if (mixed_fp16_out) mixed_fp16_out[idx] = __float2half(val);
}

// Exact IEEE-754 binary16 bitwise unpack of two signed int8s (`pair16` low 16 bits) + dual-issue FP16 `__hmul2`!
__device__ __forceinline__ __half2 i8x2_scale_h2(uint32_t pair16, __half2 bias1152, __half2 sc2) {
    uint32_t y = ((pair16 & 0x00ffu) | ((pair16 & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u; // 1024.0 + (int8 + 128) = 1152.0 + int8 (exact 1.0 ULP!)
    return __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&y), bias1152), sc2);
}

__device__ __forceinline__ void dequant_kv32_int8_to_fp16(const signed char* src32, unsigned short sc_bits, __half* dst32) {
    const uint4 v0 = *(const uint4*) src32;
    const uint4 v1 = *(const uint4*) (src32 + 16);
    const uint32_t b1152 = 0x64806480u;
    const __half2 bias1152 = *reinterpret_cast<const __half2*>(&b1152);
    const uint32_t s32 = (uint32_t) sc_bits | ((uint32_t) sc_bits << 16);
    const __half2 sc2 = *reinterpret_cast<const __half2*>(&s32);
    __half2* d2 = (__half2*) dst32;
    d2[0]  = i8x2_scale_h2(v0.x,       bias1152, sc2);
    d2[1]  = i8x2_scale_h2(v0.x >> 16, bias1152, sc2);
    d2[2]  = i8x2_scale_h2(v0.y,       bias1152, sc2);
    d2[3]  = i8x2_scale_h2(v0.y >> 16, bias1152, sc2);
    d2[4]  = i8x2_scale_h2(v0.z,       bias1152, sc2);
    d2[5]  = i8x2_scale_h2(v0.z >> 16, bias1152, sc2);
    d2[6]  = i8x2_scale_h2(v0.w,       bias1152, sc2);
    d2[7]  = i8x2_scale_h2(v0.w >> 16, bias1152, sc2);
    d2[8]  = i8x2_scale_h2(v1.x,       bias1152, sc2);
    d2[9]  = i8x2_scale_h2(v1.x >> 16, bias1152, sc2);
    d2[10] = i8x2_scale_h2(v1.y,       bias1152, sc2);
    d2[11] = i8x2_scale_h2(v1.y >> 16, bias1152, sc2);
    d2[12] = i8x2_scale_h2(v1.z,       bias1152, sc2);
    d2[13] = i8x2_scale_h2(v1.z >> 16, bias1152, sc2);
    d2[14] = i8x2_scale_h2(v1.w,       bias1152, sc2);
    d2[15] = i8x2_scale_h2(v1.w >> 16, bias1152, sc2);
}

// SM75 WMMA Tensor-Core GQA-12 Sparse Attention (`TILE_KV = 32` keys/step, 26.2 KB static shared memory -> 2 blocks/SM = 50% occupancy!)
// Grid: dim3(KVH = 2, num_steps), Block: 256 (8 warps)
__global__ void __launch_bounds__(256, 2) pf_qsa_wmma_sparse_attn(
    const float* __restrict__ qg,
    const signed char* __restrict__ vk_q, const unsigned short* __restrict__ vk_s,
    const signed char* __restrict__ vv_q, const unsigned short* __restrict__ vv_s,
    const signed char* __restrict__ hk_q, const unsigned short* __restrict__ hk_s,
    const signed char* __restrict__ hv_q, const unsigned short* __restrict__ hv_s,
    const int* __restrict__ sel_ids, const int* __restrict__ base_pos_dev,
    __half* __restrict__ out_fp16, int ring_cap, float* __restrict__ out_f32 = nullptr) {

    using namespace nvcuda;
    __shared__ __half q_tile[16][264];
    __shared__ union {
        __half kv_tile[32][264];
        float out_tile[16][264];
        float score_part[4][16][36];
    } u_kv;
    __shared__ __half prob_tile[16][40];
    __shared__ float row_m[16];
    __shared__ float row_l[16];
    __shared__ float row_alpha[16];

    const int kv_head = blockIdx.x;
    const int qi = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h_base = kv_head * 12;
    const int head_off = kv_head * HD;
    const int sc_off = kv_head * (HD / KV_GRP);

    const int cur_pos = (*base_pos_dev) + qi;
    const int n_kv = cur_pos + 1;
    const int width = min(n_kv, IDX_TOPK);
    const int max_written_n_kv = max(n_kv, (*base_pos_dev) + (int) gridDim.y);
    const int* ids = sel_ids + (size_t) qi * IDX_TOPK;

    #pragma unroll
    for (int h = 0; h < 12; ++h) {
        q_tile[h][tid] = __float2half(qg[((size_t) qi * QH + (h_base + h)) * (2 * HD) + tid] * 0.0625f);
    }
    #pragma unroll
    for (int h = 12; h < 16; ++h) {
        q_tile[h][tid] = __float2half(0.0f);
    }
    if (tid < 16) {
        row_m[tid] = -1e30f;
        row_l[tid] = 0.0f;
        row_alpha[tid] = 1.0f;
    }
    __syncthreads();

    float reg_out[12];
    #pragma unroll
    for (int h = 0; h < 12; ++h) reg_out[h] = 0.0f;

    const int ld_cell = tid >> 3;       // 0..31
    const int ld_d0   = (tid & 7) * 32; // 0..224 (32 contiguous bytes per thread!)
    const int ld_grp  = ld_d0 >> 6;     // 0..3

    const int s_nhalf = warp & 1;       // 0..1 (keys 0..15 vs 16..31)
    const int s_kpart = warp >> 1;      // 0..3 (each warp covers 64 dims of 256!)

    for (int base_idx = 0; base_idx < width; base_idx += 32) {
        const int step_cells = min(32, width - base_idx);

        // 1. Coalesced load & dequantize 32 INT8 K-rows into `u_kv.kv_tile[32][264]`
        if (ld_cell < step_cells) {
            const int idx = base_idx + ld_cell;
            const int p = (n_kv <= IDX_TOPK) ? idx : ids[idx];
            const bool in_vram = kv_is_in_vram(p, max_written_n_kv, ring_cap, hk_q);
            const size_t vslot = kv_vram_slot(p, ring_cap);
            const signed char* kp = in_vram
                ? (vk_q + vslot * KVD + head_off)
                : (hk_q + (size_t) p * KVD + head_off);
            const unsigned short* ks = in_vram
                ? (vk_s + vslot * KV_SC_PER_CELL + sc_off)
                : (hk_s + (size_t) p * KV_SC_PER_CELL + sc_off);
            dequant_kv32_int8_to_fp16(kp + ld_d0, ks[ld_grp], &u_kv.kv_tile[ld_cell][ld_d0]);
        } else {
            uint4* dst4 = (uint4*) (&u_kv.kv_tile[ld_cell][ld_d0]);
            #pragma unroll
            for (int u = 0; u < 4; ++u) dst4[u] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // 2. Tensor-Core `Q [16, 256] x K^T [256, 32] -> score_part [4][16][32]`
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
            wmma::fill_fragment(acc_s, 0.0f);
            const int k_base = s_kpart * 64;
            #pragma unroll
            for (int ks = 0; ks < 64; ks += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, &q_tile[0][k_base + ks], 264);
                wmma::load_matrix_sync(b_frag, &u_kv.kv_tile[s_nhalf * 16][k_base + ks], 264);
                wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
            }
            __syncthreads();
            wmma::store_matrix_sync(&u_kv.score_part[s_kpart][0][s_nhalf * 16], acc_s, 36, wmma::mem_row_major);
        }
        __syncthreads();

        // 3. Warp-parallel Online Softmax across 32 keys (`lane = 0..31`) for heads `0..11`
        for (int h = warp; h < 16; h += 8) {
            if (h < 12) {
                const float sv = (lane < step_cells)
                    ? (u_kv.score_part[0][h][lane] + u_kv.score_part[1][h][lane]
                     + u_kv.score_part[2][h][lane] + u_kv.score_part[3][h][lane])
                    : -1e30f;
                float rmax = sv;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                const float m_old = row_m[h];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = __expf(m_old - m_new);
                const float pw = (lane < step_cells) ? __expf(sv - m_new) : 0.0f;
                const float psum = dwsum(pw);
                if (lane == 0) {
                    row_m[h] = m_new;
                    row_l[h] = row_l[h] * alpha + psum;
                    row_alpha[h] = alpha;
                }
                prob_tile[h][lane] = __float2half(pw);
            } else {
                prob_tile[h][lane] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // 4. Coalesced load & dequantize 32 INT8 V-rows into `u_kv.kv_tile[32][264]` (`row_major` `[32, 256]`)
        if (ld_cell < step_cells) {
            const int idx = base_idx + ld_cell;
            const int p = (n_kv <= IDX_TOPK) ? idx : ids[idx];
            const bool in_vram = kv_is_in_vram(p, max_written_n_kv, ring_cap, hv_q);
            const size_t vslot = kv_vram_slot(p, ring_cap);
            const signed char* vp = in_vram
                ? (vv_q + vslot * KVD + head_off)
                : (hv_q + (size_t) p * KVD + head_off);
            const unsigned short* vs = in_vram
                ? (vv_s + vslot * KV_SC_PER_CELL + sc_off)
                : (hv_s + (size_t) p * KV_SC_PER_CELL + sc_off);
            dequant_kv32_int8_to_fp16(vp + ld_d0, vs[ld_grp], &u_kv.kv_tile[ld_cell][ld_d0]);
        } else {
            uint4* dst4 = (uint4*) (&u_kv.kv_tile[ld_cell][ld_d0]);
            #pragma unroll
            for (int u = 0; u < 4; ++u) dst4[u] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // 5. Tensor-Core `P [16, 32] x V [32, 256] -> out_tile [16, 256]` (each warp owns 2 x 16 = 32 output columns!)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_f0, p_f1;
            wmma::load_matrix_sync(p_f0, &prob_tile[0][0], 40);
            wmma::load_matrix_sync(p_f1, &prob_tile[0][16], 40);

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_v0, acc_v1;
            wmma::fill_fragment(acc_v0, 0.0f);
            wmma::fill_fragment(acc_v1, 0.0f);

            const int d_col0 = warp * 32;
            const int d_col1 = warp * 32 + 16;
            {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_f0, v_f1;
                wmma::load_matrix_sync(v_f0, &u_kv.kv_tile[0][d_col0], 264);
                wmma::load_matrix_sync(v_f1, &u_kv.kv_tile[16][d_col0], 264);
                wmma::mma_sync(acc_v0, p_f0, v_f0, acc_v0);
                wmma::mma_sync(acc_v0, p_f1, v_f1, acc_v0);
            }
            {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_f0, v_f1;
                wmma::load_matrix_sync(v_f0, &u_kv.kv_tile[0][d_col1], 264);
                wmma::load_matrix_sync(v_f1, &u_kv.kv_tile[16][d_col1], 264);
                wmma::mma_sync(acc_v1, p_f0, v_f0, acc_v1);
                wmma::mma_sync(acc_v1, p_f1, v_f1, acc_v1);
            }
            __syncthreads();
            wmma::store_matrix_sync(&u_kv.out_tile[0][d_col0], acc_v0, 264, wmma::mem_row_major);
            wmma::store_matrix_sync(&u_kv.out_tile[0][d_col1], acc_v1, 264, wmma::mem_row_major);
        }
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < 12; ++h) {
            reg_out[h] = fmaf(reg_out[h], row_alpha[h], u_kv.out_tile[h][tid]);
        }
        __syncthreads();
    }

    if (tid < 12) row_alpha[tid] = 1.0f / fmaxf(row_l[tid], 1e-20f);
    __syncthreads();

    #pragma unroll
    for (int h = 0; h < 12; ++h) {
        const float gate = qg[((size_t) qi * QH + (h_base + h)) * (2 * HD) + HD + tid];
        const float val = (reg_out[h] * row_alpha[h]) / (1.0f + __expf(-gate));
        if (out_f32) out_f32[((size_t) qi * QH + (h_base + h)) * HD + tid] = val;
        if (out_fp16) out_fp16[((size_t) qi * QH + (h_base + h)) * HD + tid] = __float2half(val);
    }
}

// SM75 WMMA Tensor-Core QSA Sparse Block Scorer (`[16*4, 128] x [64, 128]^T` -> 64x shared-memory reuse, ZERO warp shuffles!)
// Grid: dim3((max_blocks + 63) / 64, (qb + 15) / 16), Block: 256 (8 warps)
__global__ void __launch_bounds__(256) pf_qsa_block_scores_wmma(
    const float* __restrict__ pooled,
    const float* __restrict__ q_idx,
    const int* __restrict__ base_pos_dev,
    int max_blocks, float* __restrict__ scores_out, int qi_off, int qb) {

    using namespace nvcuda;
    const int base_pos = *base_pos_dev;
    const int q_local_max = min(qb - 1, (int) blockIdx.y * 16 + 15);
    const int max_n_kv = base_pos + qi_off + q_local_max + 1;
    if (max_n_kv <= IDX_TOPK) return;
    const int b0 = blockIdx.x * 64;
    if (b0 >= (max_n_kv / IDX_BLK)) return; // causal upper-triangle skip!

    __shared__ union {
        struct {
            __half Q[64][136];
            __half K[64][136];
        } stg;
        float C[64][68];
    } smem;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_m = warp >> 1; // 0..3 (16 query-head rows each)
    const int warp_n = warp & 1;  // 0..1 (32 key-block cols each = 2 x 16)

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx4 = p * 256 + tid; // 0..2047
        const int row = idx4 >> 5;      // 0..63
        const int d4 = (idx4 & 31) * 4; // 0..124
        const int q_local = blockIdx.y * 16 + (row >> 2);
        const int h = row & 3;
        if (q_local < qb) {
            const float4 qv = *(const float4*) (q_idx + ((size_t) (qi_off + q_local) * IDX_HEADS + h) * IDX_DIM + d4);
            *(__half2*) (&smem.stg.Q[row][d4 + 0]) = __floats2half2_rn(qv.x, qv.y);
            *(__half2*) (&smem.stg.Q[row][d4 + 2]) = __floats2half2_rn(qv.z, qv.w);
        } else {
            *(uint2*) (&smem.stg.Q[row][d4]) = make_uint2(0, 0);
        }
        const int b = b0 + row;
        if (b < max_blocks) {
            const float4 kv = *(const float4*) (pooled + (size_t) b * IDX_DIM + d4);
            *(__half2*) (&smem.stg.K[row][d4 + 0]) = __floats2half2_rn(kv.x, kv.y);
            *(__half2*) (&smem.stg.K[row][d4 + 2]) = __floats2half2_rn(kv.z, kv.w);
        } else {
            *(uint2*) (&smem.stg.K[row][d4]) = make_uint2(0, 0);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    #pragma unroll
    for (int ks = 0; ks < IDX_DIM; ks += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag0, b_frag1;
        wmma::load_matrix_sync(a_frag, &smem.stg.Q[warp_m * 16][ks], 136);
        wmma::load_matrix_sync(b_frag0, &smem.stg.K[warp_n * 32][ks], 136);
        wmma::load_matrix_sync(b_frag1, &smem.stg.K[warp_n * 32 + 16][ks], 136);
        wmma::mma_sync(acc0, a_frag, b_frag0, acc0);
        wmma::mma_sync(acc1, a_frag, b_frag1, acc1);
    }
    __syncthreads();
    wmma::store_matrix_sync(&smem.C[warp_m * 16][warp_n * 32], acc0, 68, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem.C[warp_m * 16][warp_n * 32 + 16], acc1, 68, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int pair_idx = u * 256 + tid; // 0..1023
        const int q_sub = pair_idx >> 6;    // 0..15
        const int bk = pair_idx & 63;       // 0..63
        const int q_local = blockIdx.y * 16 + q_sub;
        const int b = b0 + bk;
        if (q_local < qb && b < max_blocks) {
            const int n_kv = base_pos + qi_off + q_local + 1;
            if (n_kv > IDX_TOPK && b < (n_kv / IDX_BLK)) {
                const float d0 = smem.C[q_sub * 4 + 0][bk];
                const float d1 = smem.C[q_sub * 4 + 1][bk];
                const float d2 = smem.C[q_sub * 4 + 2][bk];
                const float d3 = smem.C[q_sub * 4 + 3][bk];
                const float sc = (d0 > 0.f ? d0 : 0.f) + (d1 > 0.f ? d1 : 0.f)
                               + (d2 > 0.f ? d2 : 0.f) + (d3 > 0.f ? d3 : 0.f);
                scores_out[(size_t) q_local * max_blocks + b] = sc;
            }
        }
    }
}

// Exact tail-block (`b == n_bid`) scorer against `dead` (+ 1e9f partial-block boost)
// Grid: qb, Block: 128 (4 warps = 4 indexer heads)
__global__ void pf_qsa_tail_score_kernel(
    const float* __restrict__ dead,
    const float* __restrict__ q_idx,
    const int* __restrict__ base_pos_dev,
    int max_blocks, float* __restrict__ scores_out, int qi_off) {

    const int q_local = blockIdx.x;
    const int qi = qi_off + q_local;
    const int n_kv = (*base_pos_dev) + qi + 1;
    if (n_kv <= IDX_TOPK) return;
    const int n_bid = n_kv / IDX_BLK;
    if (n_bid >= max_blocks) return;

    __shared__ float s_relu[4];
    const int lane = threadIdx.x & 31;
    const int h = threadIdx.x >> 5; // 0..3
    const float4 k4 = *(const float4*) (dead + lane * 4);
    const float4 q4 = *(const float4*) (q_idx + ((size_t) qi * IDX_HEADS + h) * IDX_DIM + lane * 4);
    const float d = dwsum(k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w);
    if (lane == 0) s_relu[h] = d > 0.f ? d : 0.f;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sc = s_relu[0] + s_relu[1] + s_relu[2] + s_relu[3];
        if ((n_kv % IDX_BLK) != 0) sc += 1e9f;
        scores_out[(size_t) q_local * max_blocks + n_bid] = sc;
    }
}
