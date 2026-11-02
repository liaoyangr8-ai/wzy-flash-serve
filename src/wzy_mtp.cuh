// wzy_mtp.cuh -- Native SM75 MTP Speculative Drafter (<MODEL>/mtp Q2_0 Experts + 114,452 CJK Draft Vocab)
#pragma once
#include "wzy_stage_v2.cuh"
#include <sstream>

static constexpr size_t MTP_Q2_BLOB = 1382400; // 3 * (2560 * 640 * 18 / 64)
static constexpr size_t MTP_O_GU_CODES = 0;
static constexpr size_t MTP_O_D_CODES  = 2ull * NFF * (NE / 4);                  // 819,200
static constexpr size_t MTP_O_GU_SCALES = MTP_O_D_CODES + 1ull * NE * (NFF / 4); // 1,228,800
static constexpr size_t MTP_O_D_SCALES  = MTP_O_GU_SCALES + 2ull * NFF * (NE / 64) * 2; // 1,331,200
static constexpr int MTP_KV_RING = 1024;  // 1024-cell INT8 KV ring (4 KB shared memory -> full occupancy + 1K window!)
static constexpr int MTP_SEED_TAIL = 512; // seed at most the last 512 prompt cells (~9 ms) during long prefill!

// mtp_gather_bf16_emb now lives in wzy_kernels_v2.cuh (stage_v2 needs it too).

// Per-row weighted RMSNorm: out[r, d] = in[r, d] * rsqrt(mean(in[r]^2) + eps) * gamma[d % g_mod]
__global__ void mtp_rms_norm_rows(const float* __restrict__ in, const float* __restrict__ gamma,
                                  float* __restrict__ out, int cols, int g_stride_rows, float eps) {
    const int r = blockIdx.x;
    const int g_row = r % g_stride_rows;
    const float* x = in + (size_t) r * cols;
    const float* g = gamma + (size_t) g_row * cols;
    float* y = out + (size_t) r * cols;

    float ss = 0.f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) ss += x[d] * x[d];
    ss = dwsum(ss);
    __shared__ float sh[8];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) tot += sh[i];
        sh[0] = rsqrtf(tot / (float) cols + eps);
    }
    __syncthreads();
    const float sc = sh[0];
    for (int d = threadIdx.x; d < cols; d += blockDim.x) y[d] = x[d] * sc * g[d];
}

// R_out[t, c, d] = h2[t, c, d] + e2[t, d]
__global__ void mtp_add_streams_bcast(const float* __restrict__ h2, const float* __restrict__ e2,
                                      float* __restrict__ R_out, int n_embd, int hc, int T) {
    const int total = T * hc * n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        const int rem = i % (hc * n_embd);
        const int t = i / (hc * n_embd);
        const int d = rem % n_embd;
        R_out[i] = h2[i] + e2[(size_t) t * n_embd + d];
    }
}

// Q2_0 Routed Experts (all 512 experts 100% resident in GPU1 VRAM! Exact match to s2_expert_grouped.cu)
__device__ __forceinline__ float mtp_f16_at(const uint8_t* p) {
    unsigned short u = (unsigned short) (p[0] | (p[1] << 8));
    return dh2f(u);
}

__device__ __forceinline__ float mtp_row_dot_q2(const uint8_t* __restrict__ codes,
                                                const uint8_t* __restrict__ scales,
                                                const signed char* __restrict__ xq32,
                                                const float* __restrict__ xs32,
                                                int n_chunks, int lane) {
    float acc = 0.f;
    const int ones = 0x01010101;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;
        const int* xw = (const int*) (xq32 + (size_t) c * 32);
        int s = 0, hx = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned b = cb[j];
            const int cw = (int) ((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) | (((b >> 6) & 3u) << 24));
            const int xv = xw[j];
            s = __dp4a(cw, xv, s);
            hx = __dp4a(ones, xv, hx);
        }
        const float dw = mtp_f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * xs32[c] * (float) (s - hx);
    }
    return acc;
}

// 2-rows-per-warp Q2_0 GU + fused INT8 block-32 quantization (16 warps per block = 32 rows = 1 quant block!)
__global__ void mtp_exp_gu_quant_q2(const uint8_t* __restrict__ experts_vram, const int* __restrict__ ids, int total_k,
                                    const signed char* __restrict__ xq32, const float* __restrict__ xs32,
                                    signed char* __restrict__ dq32, float* __restrict__ ds32) {
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_in_blk = threadIdx.x >> 5;
    const int e = gwarp / (NFF / 2);
    if (e >= total_k) return;
    const int row0 = (gwarp - e * (NFF / 2)) * 2;
    const int t = e / TOPK;
    const int eid = ids[e];

    const uint8_t* blob = experts_vram + (size_t) eid * MTP_Q2_BLOB;
    const uint8_t* g0_c = blob + MTP_O_GU_CODES + (size_t) (2 * row0) * (NE / 4);
    const uint8_t* u0_c = blob + MTP_O_GU_CODES + (size_t) (2 * row0 + 1) * (NE / 4);
    const uint8_t* g1_c = blob + MTP_O_GU_CODES + (size_t) (2 * row0 + 2) * (NE / 4);
    const uint8_t* u1_c = blob + MTP_O_GU_CODES + (size_t) (2 * row0 + 3) * (NE / 4);

    const uint8_t* g0_s = blob + MTP_O_GU_SCALES + (size_t) (2 * row0) * (NE / 64) * 2;
    const uint8_t* u0_s = blob + MTP_O_GU_SCALES + (size_t) (2 * row0 + 1) * (NE / 64) * 2;
    const uint8_t* g1_s = blob + MTP_O_GU_SCALES + (size_t) (2 * row0 + 2) * (NE / 64) * 2;
    const uint8_t* u1_s = blob + MTP_O_GU_SCALES + (size_t) (2 * row0 + 3) * (NE / 64) * 2;

    const signed char* xqt = xq32 + (size_t) t * NE;
    const float* xst = xs32 + (size_t) t * (NE / 32);

    float ag0 = 0.f, au0 = 0.f, ag1 = 0.f, au1 = 0.f;
    const int ones = 0x01010101;
    for (int c = lane; c < NE / 32; c += 32) {
        const int* xw = (const int*) (xqt + (size_t) c * 32);
        const int xv0 = xw[0], xv1 = xw[1], xv2 = xw[2], xv3 = xw[3],
                  xv4 = xw[4], xv5 = xw[5], xv6 = xw[6], xv7 = xw[7];
        int hx = __dp4a(ones, xv0, 0); hx = __dp4a(ones, xv1, hx); hx = __dp4a(ones, xv2, hx); hx = __dp4a(ones, xv3, hx);
            hx = __dp4a(ones, xv4, hx); hx = __dp4a(ones, xv5, hx); hx = __dp4a(ones, xv6, hx); hx = __dp4a(ones, xv7, hx);
        const float xsc = xst[c];
        auto dot8 = [&](const uint8_t* cb) __attribute__((always_inline)) {
            int s = 0;
            const int xv_arr[8] = {xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned b = cb[j];
                const int cw = (int) ((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) | (((b >> 6) & 3u) << 24));
                s = __dp4a(cw, xv_arr[j], s);
            }
            return (float) (s - hx);
        };
        const size_t co = (size_t) c * 8;
        const size_t so = (size_t) (c >> 1) * 2;
        ag0 += mtp_f16_at(g0_s + so) * xsc * dot8(g0_c + co);
        au0 += mtp_f16_at(u0_s + so) * xsc * dot8(u0_c + co);
        ag1 += mtp_f16_at(g1_s + so) * xsc * dot8(g1_c + co);
        au1 += mtp_f16_at(u1_s + so) * xsc * dot8(u1_c + co);
    }
    ag0 = dwsum(ag0); au0 = dwsum(au0);
    ag1 = dwsum(ag1); au1 = dwsum(au1);

    __shared__ float sh_act[32];
    if (lane == 0) {
        sh_act[warp_in_blk * 2 + 0] = (ag0 / (1.0f + __expf(-ag0))) * au0;
        sh_act[warp_in_blk * 2 + 1] = (ag1 / (1.0f + __expf(-ag1))) * au1;
    }
    __syncthreads();
    if (warp_in_blk == 0) {
        const float v = sh_act[lane];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float sc = amax / 127.0f;
        const float inv = amax > 0.f ? 127.0f / amax : 0.f;
        const size_t blk32 = (size_t) e * (NFF / 32) + (size_t) (row0 >> 5);
        dq32[blk32 * 32 + lane] = (signed char) __float2int_rn(v * inv);
        if (lane == 0) ds32[blk32] = sc;
    }
}

__global__ void mtp_exp_down_2r_q2(const uint8_t* __restrict__ experts_vram, const int* __restrict__ ids, int total_k,
                                   const signed char* __restrict__ xq32, const float* __restrict__ xs32,
                                   float* __restrict__ out) {
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int e = gwarp / (NE / 2);
    if (e >= total_k) return;
    const int row0 = (gwarp - e * (NE / 2)) * 2;
    const int eid = ids[e];

    const uint8_t* blob = experts_vram + (size_t) eid * MTP_Q2_BLOB;
    const uint8_t* d0_c = blob + MTP_O_D_CODES + (size_t) row0 * (NFF / 4);
    const uint8_t* d1_c = blob + MTP_O_D_CODES + (size_t) (row0 + 1) * (NFF / 4);
    const uint8_t* d0_s = blob + MTP_O_D_SCALES + (size_t) row0 * (NFF / 64) * 2;
    const uint8_t* d1_s = blob + MTP_O_D_SCALES + (size_t) (row0 + 1) * (NFF / 64) * 2;

    const signed char* xqe = xq32 + (size_t) e * NFF;
    const float* xse = xs32 + (size_t) e * (NFF / 32);

    float acc0 = 0.f, acc1 = 0.f;
    if (lane < NFF / 32) {
        const int* xw = (const int*) (xqe + (size_t) lane * 32);
        const int ones = 0x01010101;
        int s0 = 0, s1 = 0, hx = 0;
        const uint8_t* cb0 = d0_c + (size_t) lane * 8;
        const uint8_t* cb1 = d1_c + (size_t) lane * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int xv = xw[j];
            hx = __dp4a(ones, xv, hx);
            const unsigned b0 = cb0[j], b1 = cb1[j];
            const int cw0 = (int) ((b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24));
            const int cw1 = (int) ((b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24));
            s0 = __dp4a(cw0, xv, s0);
            s1 = __dp4a(cw1, xv, s1);
        }
        const float xsc = xse[lane];
        const size_t so = (size_t) (lane >> 1) * 2;
        acc0 = mtp_f16_at(d0_s + so) * xsc * (float) (s0 - hx);
        acc1 = mtp_f16_at(d1_s + so) * xsc * (float) (s1 - hx);
    }
    acc0 = dwsum(acc0);
    acc1 = dwsum(acc1);
    if (lane == 0) {
        out[(size_t) e * NE + row0 + 0] = acc0;
        out[(size_t) e * NE + row0 + 1] = acc1;
    }
}

// Dense sliding-window INT8 QSA attention for MTP Drafter (attends to `max(0, n_kv - MTP_KV_RING) .. n_kv - 1`)
// Grid: QH (24), Block: 256 (single row: row 0)
__global__ void mtp_qsa_attend_ring(const float* __restrict__ qg,
                                    const signed char* __restrict__ vk_q, const unsigned short* __restrict__ vk_s,
                                    const signed char* __restrict__ vv_q, const unsigned short* __restrict__ vv_s,
                                    const int* __restrict__ pos_dev, const int* __restrict__ kv_floor_dev,
                                    float* __restrict__ out) {
    __shared__ float sc[MTP_KV_RING]; // 16 KB shared memory!
    __shared__ float qsh[HD];
    __shared__ float wr[8];

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int kv_head = h / (QH / KVH);
    const int head_off = kv_head * HD;
    const int sc_off = kv_head * (HD / KV_GRP);

    const int cur_pos = *pos_dev;
    const int n_kv = cur_pos + 1;
    const int p_start = max(kv_floor_dev ? *kv_floor_dev : 0, max(0, n_kv - MTP_KV_RING));
    const int width = n_kv - p_start;

    const float* qrow = qg + (size_t) h * (2 * HD);
    qsh[tid] = qrow[tid];
    __syncthreads();

    const int d0 = lane * 8;
    const int g_lane = lane >> 3;
    const float4 qf0 = *(const float4*) (qsh + d0);
    const float4 qf1 = *(const float4*) (qsh + d0 + 4);
    float local_max = -1e30f;
    for (int idx = warp; idx < width; idx += 8) {
        const int p = p_start + idx;
        const size_t rslot = (size_t) (p % MTP_KV_RING);
        const signed char* kp = vk_q + rslot * KVD + head_off;
        const unsigned short* ks = vk_s + rslot * KV_SC_PER_CELL + sc_off;
        const char4 kv0 = *(const char4*) (kp + d0);
        const char4 kv1 = *(const char4*) (kp + d0 + 4);
        const float part = dh2f(ks[g_lane]) * (
              (float) kv0.x * qf0.x + (float) kv0.y * qf0.y + (float) kv0.z * qf0.z + (float) kv0.w * qf0.w
            + (float) kv1.x * qf1.x + (float) kv1.y * qf1.y + (float) kv1.z * qf1.z + (float) kv1.w * qf1.w);
        const float s = dwsum(part) * 0.0625f;
        if (lane == 0) {
            sc[idx] = s;
            local_max = fmaxf(local_max, s);
        }
    }

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

    const int d = tid, vg = d >> 6;
    float acc = 0.f;
    for (int idx = 0; idx < width; ++idx) {
        const size_t rslot = (size_t) ((p_start + idx) % MTP_KV_RING);
        const signed char vcode = vv_q[rslot * KVD + head_off + d];
        const unsigned short vsh = vv_s[rslot * KV_SC_PER_CELL + sc_off + vg];
        acc += sc[idx] * (dh2f(vsh) * (float) vcode);
    }
    acc *= inv_sum;
    out[h * HD + d] = acc / (1.0f + __expf(-qrow[HD + d]));
}

// Two-stage GPU argmax + softmax top-1 probability over `n_dvocab` (114,452) logits -> zero D2H copy of logits!
__global__ void mtp_head_reduce_online_part1(const float* __restrict__ logits, int n,
                                             float* __restrict__ blk_max, int* __restrict__ blk_idx,
                                             float* __restrict__ blk_sum) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    float bv = -1e30f;
    int bi = 0;
    float es = 0.f;
    for (int i = bid * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        const float v = logits[i];
        if (v > bv) {
            es = es * __expf(bv - v) + 1.0f;
            bv = v;
            bi = i;
        } else {
            es += __expf(v - bv);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        const float os = __shfl_xor_sync(0xffffffffu, es, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            es = es * __expf(bv - ov) + os;
            bv = ov;
            bi = oi;
        } else {
            es += os * __expf(ov - bv);
        }
    }
    __shared__ float sv[8], ss[8];
    __shared__ int si[8];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; ss[warp] = es; }
    __syncthreads();
    if (warp == 0) {
        bv = (lane < 8) ? sv[lane] : -1e30f;
        bi = (lane < 8) ? si[lane] : 0;
        es = (lane < 8) ? ss[lane] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, es, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                es = es * __expf(bv - ov) + os;
                bv = ov;
                bi = oi;
            } else {
                es += os * __expf(ov - bv);
            }
        }
        if (lane == 0) { blk_max[bid] = bv; blk_idx[bid] = bi; blk_sum[bid] = es; }
    }
}

__global__ void mtp_head_finalize(const float* __restrict__ blk_max, const int* __restrict__ blk_idx,
                                  const float* __restrict__ blk_sum, int n_blks,
                                  const int* __restrict__ dvocab,
                                  int* __restrict__ out_tok_id, float* __restrict__ out_prob) {
    const int tid = threadIdx.x; // 128 threads, n_blks == 128
    float bv = (tid < n_blks) ? blk_max[tid] : -1e30f;
    int bi = (tid < n_blks) ? blk_idx[tid] : 0;
    float s = (tid < n_blks) ? blk_sum[tid] : 0.f;

    const int lane = tid & 31, warp = tid >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        const float os = __shfl_xor_sync(0xffffffffu, s, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            s = s * __expf(bv - ov) + os;
            bv = ov;
            bi = oi;
        } else {
            s += os * __expf(ov - bv);
        }
    }
    __shared__ float sv[4], ss[4];
    __shared__ int si[4];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; ss[warp] = s; }
    __syncthreads();
    if (tid == 0) {
        float fmax = sv[0]; int fidx = si[0]; float fsum = ss[0];
        for (int w = 1; w < 4; ++w) {
            const float ov = sv[w];
            const int oi = si[w];
            const float os = ss[w];
            if (ov > fmax || (ov == fmax && oi < fidx)) {
                fsum = fsum * __expf(fmax - ov) + os;
                fmax = ov;
                fidx = oi;
            } else {
                fsum += os * __expf(ov - fmax);
            }
        }
        *out_tok_id = dvocab ? dvocab[fidx] : fidx;
        *out_prob = 1.0f / fmaxf(fsum, 1.0f);
    }
}

struct MtpEngine {
    bool enabled = false;
    int dev = 1;
    cudaStream_t stream = nullptr;
    uint16_t* d_bf_emb = nullptr; // device BF16 embedding table [NVOCAB, NE] (1.21 GiB? No: gather on host or keep Q8/mapped!)

    // Dense MTP weights
    Q8 fc_emb, fc_hid, q_proj, k_proj, v_proj, o_proj, sh_gate, sh_up, sh_down;
    Dev pre_norm_emb, pre_norm_hid, q_norm, k_norm;
    Dev ahc_norm, mhc_norm, mix_norm;
    Q8 ahc_down, ahc_up, mhc_down, mhc_up, mix_down, mix_up;
    Bf ahc_inj, mhc_inj;
    Bf router_w, sh_gate_inp;

    // 512 resident Q2_0 experts (675 MiB on GPU 1!)
    uint8_t* d_experts = nullptr;

    // UVA (mapped host) BF16 token embedding table: lets the draft chain gather the next token's
    // embedding ON DEVICE, so k chained MTP steps need one host sync instead of k.
    const uint16_t* d_emb_table = nullptr;

    void set_emb_table(const uint16_t* t) { d_emb_table = t; }
    bool emb_gather_ready() const { return d_emb_table != nullptr; }

    // Fills `emb` [T, NE] from device tokens `d_toks` (device int array). No host roundtrip.
    void gather_emb_dev(const int* d_toks, int T, cudaStream_t s) {
        if (d_emb_table == nullptr) return;
        dim3 grid((NE + 255) / 256, T);
        mtp_gather_bf16_emb<<<grid, 256, 0, s>>>(d_emb_table, d_toks, emb.p, T);
    }

    // Draft head subset (114,452 rows of output.weight = 297 MiB on GPU 1!)
    int n_dvocab = 0;
    Q8 d_head;
    int* d_dvocab = nullptr;

    // Sliding-window INT8 KV ring (4096 cells = 4.3 MiB!)
    signed char *vk_q = nullptr, *vv_q = nullptr;
    unsigned short *vk_s = nullptr, *vv_s = nullptr;

    // Scratch for up to MAX_MTP_B = 64 catch-up cells & 1-step draft chain
    static constexpr int MAX_B = 64;
    Dev emb, en, e2, Rin, hn, h2, R_cur;
    Dev xn, lo, mixed, inj, inj2, qg, kk, vv, attn, bo;
    Dev slog, sids, swts, sff, sup, tmp_ff, sshared, sgate, ffact, parts, comb, sample, head_logits;
    signed char *xq32 = nullptr, *dq32 = nullptr;
    float *xs32 = nullptr, *ds32 = nullptr;
    int *d_pos = nullptr, *h_pos = nullptr;
    int *d_kv_floor = nullptr;
    float *h_emb_stage = nullptr;
    float *d_blk_max = nullptr, *d_blk_sum = nullptr, *d_best_logit = nullptr;
    int *d_blk_idx = nullptr;
    int *d_out_tok = nullptr, *h_out_tok = nullptr;
    float *d_out_prob = nullptr, *h_out_prob = nullptr;

    cudaGraphExec_t step1_exec = nullptr; // captured 1-row full draft step!

    bool init(int gpu_dev, const Gguf& main_gguf, const std::string& mtp_dir, const std::string& dvocab_path,
              cudaStream_t st) {
        dev = gpu_dev;
        stream = st;
        CHECK(cudaSetDevice(dev));

        std::ifstream idx(mtp_dir + "/dense.txt");
        if (!idx) { std::fprintf(stderr, "[mtp] cannot open %s/dense.txt\n", mtp_dir.c_str()); return false; }
        struct MT { std::string kind; int rows, cols; size_t off, bytes; };
        std::map<std::string, MT> mp;
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            std::string nm; MT m;
            if (is >> nm >> m.kind >> m.rows >> m.cols >> m.off >> m.bytes) mp[nm] = m;
        }
        std::ifstream fb(mtp_dir + "/dense.bin", std::ios::binary);
        if (!fb) return false;
        fb.seekg(0, std::ios::end);
        std::vector<uint8_t> dblob((size_t) fb.tellg());
        fb.seekg(0);
        fb.read((char*) dblob.data(), dblob.size());

        auto ld_q8 = [&](const char* nm) -> Q8 {
            const auto& m = mp.at(nm);
            return up_q8_raw(dblob.data() + m.off, m.rows, m.cols);
        };
        auto ld_bf = [&](const char* nm) -> Bf {
            const auto& m = mp.at(nm);
            Bf b; b.rows = m.rows; b.cols = m.cols;
            CHECK(cudaMalloc(&b.p, m.bytes));
            CHECK(cudaMemcpy(b.p, dblob.data() + m.off, m.bytes, cudaMemcpyHostToDevice));
            return b;
            return b;
        };
        auto ld_bf_as_q8 = [&](const char* nm) -> Q8 {
            Bf b = ld_bf(nm);
            Q8 q; q.rows = b.rows; q.cols = b.cols;
            const size_t nb32 = (size_t) q.rows * (q.cols / 32);
            CHECK(cudaMalloc(&q.p, (size_t) q.rows * q.cols));
            CHECK(cudaMalloc(&q.s, nb32 * 2));
            quant_bf16_to_q8_kernel<<<(nb32 + 7) / 8, 256>>>(b.p, q.p, q.s, nb32);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaFree(b.p));
            return q;
        };
        auto ld_f32 = [&](const char* nm) -> Dev {
            const auto& m = mp.at(nm);
            Dev d = alloc(m.bytes / 4);
            CHECK(cudaMemcpy(d.p, dblob.data() + m.off, m.bytes, cudaMemcpyHostToDevice));
            return d;
        };

        fc_emb = ld_q8("fc_embedding.weight");
        fc_hid = ld_q8("fc_hidden.weight");
        pre_norm_emb = ld_f32("pre_fc_norm_embedding.weight");
        pre_norm_hid = ld_f32("pre_fc_norm_hidden.weight");

        ahc_norm = ld_f32("attn_hyper_connection.hc_norm.weight");
        ahc_down = ld_bf_as_q8("attn_hyper_connection.input_mix_weight_down.weight");
        ahc_up   = ld_bf_as_q8("attn_hyper_connection.input_mix_weight_up.weight");
        ahc_inj  = ld_bf("attn_hyper_connection.block_inject_weight.weight");

        q_proj = ld_q8("self_attn.q_proj.weight");
        k_proj = ld_q8("self_attn.k_proj.weight");
        v_proj = ld_q8("self_attn.v_proj.weight");
        o_proj = ld_q8("self_attn.o_proj.weight");
        q_norm = ld_f32("self_attn.q_norm.weight");
        k_norm = ld_f32("self_attn.k_norm.weight");

        mhc_norm = ld_f32("mlp_hyper_connection.hc_norm.weight");
        mhc_down = ld_bf_as_q8("mlp_hyper_connection.input_mix_weight_down.weight");
        mhc_up   = ld_bf_as_q8("mlp_hyper_connection.input_mix_weight_up.weight");
        mhc_inj  = ld_bf("mlp_hyper_connection.block_inject_weight.weight");

        router_w    = ld_bf("mlp.gate.weight");
        sh_gate     = ld_q8("mlp.shared_expert.gate_proj.weight");
        sh_up       = ld_q8("mlp.shared_expert.up_proj.weight");
        sh_down     = ld_q8("mlp.shared_expert.down_proj.weight");
        sh_gate_inp = ld_bf("mlp.shared_expert_gate.weight");

        mix_norm = ld_f32("hyper_connection_mixer.hc_norm.weight");
        mix_down = ld_bf_as_q8("hyper_connection_mixer.input_mix_weight_down.weight");
        mix_up   = ld_bf_as_q8("hyper_connection_mixer.input_mix_weight_up.weight");

        // Load 512 Q2_0 experts (675 MiB) into GPU 1 VRAM!
        const size_t exp_bytes = (size_t) NEXP * MTP_Q2_BLOB;
        FILE* ef = std::fopen((mtp_dir + "/experts.bin").c_str(), "rb");
        if (!ef) { std::fprintf(stderr, "[mtp] cannot open experts.bin\n"); return false; }
        CHECK(cudaMalloc(&d_experts, exp_bytes));
        std::vector<uint8_t> chunk(64u << 20);
        for (size_t off = 0; off < exp_bytes; ) {
            const size_t n = std::min(chunk.size(), exp_bytes - off);
            if (std::fread(chunk.data(), 1, n, ef) != n) { std::fclose(ef); return false; }
            CHECK(cudaMemcpy(d_experts + off, chunk.data(), n, cudaMemcpyHostToDevice));
            off += n;
        }
        std::fclose(ef);

        // Build subset draft head over `draft_vocab_han.bin` (114,452 rows from main `output.weight`)!
        std::ifstream vf(dvocab_path, std::ios::binary);
        std::vector<int32_t> v_ids;
        if (vf) {
            vf.seekg(0, std::ios::end);
            const size_t vsz = (size_t) vf.tellg();
            vf.seekg(0);
            v_ids.resize(vsz / 4);
            vf.read((char*) v_ids.data(), vsz);
        }
        const TDesc* out_t = main_gguf.find("output.weight");
        const uint8_t* out_raw = main_gguf.at(out_t);
        const int nb = NE / 32; // 80
        const size_t row_b = (size_t) nb * 34; // 2720 B
        if (!v_ids.empty()) {
            n_dvocab = (int) v_ids.size();
            std::vector<uint8_t> sub_codes((size_t) n_dvocab * NE);
            std::vector<uint16_t> sub_scales((size_t) n_dvocab * nb);
            for (int r = 0; r < n_dvocab; ++r) {
                const int tid = std::max(0, std::min(NVOCAB - 1, (int) v_ids[r]));
                const uint8_t* w = out_raw + (size_t) tid * row_b;
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(&sub_scales[(size_t) r * nb + b], w + b * 34, 2);
                    std::memcpy(&sub_codes[(size_t) r * NE + (size_t) b * 32], w + b * 34 + 2, 32);
                }
            }
            d_head.rows = n_dvocab; d_head.cols = NE;
            CHECK(cudaMalloc(&d_head.p, sub_codes.size()));
            CHECK(cudaMalloc(&d_head.s, sub_scales.size() * 2));
            CHECK(cudaMemcpy(d_head.p, sub_codes.data(), sub_codes.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_head.s, sub_scales.data(), sub_scales.size() * 2, cudaMemcpyHostToDevice));
            CHECK(cudaMalloc(&d_dvocab, (size_t) n_dvocab * 4));
            CHECK(cudaMemcpy(d_dvocab, v_ids.data(), (size_t) n_dvocab * 4, cudaMemcpyHostToDevice));
        } else {
            return false;
        }

        // Allocate KV ring & scratch
        CHECK(cudaMalloc(&vk_q, (size_t) MTP_KV_RING * KVD));
        CHECK(cudaMalloc(&vv_q, (size_t) MTP_KV_RING * KVD));
        CHECK(cudaMalloc(&vk_s, (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2));
        CHECK(cudaMalloc(&vv_s, (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2));

        emb = alloc0(MAX_B * NE); en = alloc0(MAX_B * NE); e2 = alloc0(MAX_B * NE);
        Rin = alloc0(MAX_B * HCD); hn = alloc0(MAX_B * HCD); h2 = alloc0(MAX_B * HCD); R_cur = alloc0(MAX_B * HCD);
        xn = alloc0(MAX_B * HCD); lo = alloc0(MAX_B * HCLR); mixed = alloc0(MAX_B * NE);
        inj = alloc0(MAX_B * HC); inj2 = alloc0(MAX_B * HC);
        qg = alloc0(2 * QD); kk = alloc0(MAX_B * KVD); vv = alloc0(MAX_B * KVD);
        attn = alloc0(QD); bo = alloc0(NE);
        slog = alloc0(NEXP); sids = alloc0(TOPK); swts = alloc0(TOPK);
        sff = alloc0(NFF); sup = alloc0(NFF); tmp_ff = alloc0(NFF); sshared = alloc0(NE); sgate = alloc0(1);
        ffact = alloc0(TOPK * NFF); parts = alloc0(TOPK * NE); comb = alloc0(NE);
        sample = alloc0(NE); head_logits = alloc0(n_dvocab);

        CHECK(cudaMalloc(&xq32, MAX_B * HC * QD));
        CHECK(cudaMalloc(&xs32, (MAX_B * HC * QD / 32 + 16) * 4));
        CHECK(cudaMalloc(&dq32, TOPK * NFF));
        CHECK(cudaMalloc(&ds32, (TOPK * NFF / 32 + 8) * 4));

        CHECK(cudaMalloc(&d_pos, 4));
        CHECK(cudaHostAlloc(&h_pos, 8, cudaHostAllocPortable));
        CHECK(cudaMalloc(&d_kv_floor, 4));
        CHECK(cudaMemset(d_kv_floor, 0, 4));
        CHECK(cudaHostAlloc(&h_emb_stage, MAX_B * NE * 4, cudaHostAllocPortable));
        CHECK(cudaMalloc(&d_blk_max, 128 * 4));
        CHECK(cudaMalloc(&d_blk_sum, 128 * 4));
        CHECK(cudaMalloc(&d_best_logit, 4));
        CHECK(cudaMalloc(&d_blk_idx, 128 * 4));
        CHECK(cudaMalloc(&d_out_tok, 4));
        CHECK(cudaMalloc(&d_out_prob, 4));
        CHECK(cudaHostAlloc(&h_out_tok, 4, cudaHostAllocPortable));
        CHECK(cudaHostAlloc(&h_out_prob, 4, cudaHostAllocPortable));

        enabled = true;
        std::fprintf(stderr, "[wzy-mtp] Loaded MTP drafter on GPU%d: 512 Q2_0 experts resident + %d CJK draft vocab (%.1f MiB)\n",
                     dev, n_dvocab, (double) (n_dvocab * row_b) / 1048576.0);
        return true;
    }

    void reset_kv() {
        if (!enabled) return;
        CHECK(cudaSetDevice(dev));
        CHECK(cudaMemsetAsync(vk_q, 0, (size_t) MTP_KV_RING * KVD, stream));
        CHECK(cudaMemsetAsync(vv_q, 0, (size_t) MTP_KV_RING * KVD, stream));
        CHECK(cudaMemsetAsync(vk_s, 0, (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2, stream));
        CHECK(cudaMemsetAsync(vv_s, 0, (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2, stream));
        CHECK(cudaMemsetAsync(d_kv_floor, 0, 4, stream));
    }

    void q8_proj(const Q8& W, const float* x, float* y, int rows, int cols, int T) {
        quant_act_multi<<<((cols * T / 16) + 127) / 128, 128, 0, stream>>>(x, xq32, xs32, nullptr, nullptr, cols, T);
        launch_gemm_q8p(W.p, W.s, xq32, xs32, y, rows, cols, T, stream);
    }

    // Front half (K/V append for T rows, starting at cell `base_pos`): used for both prompt prefill & window catch-up!
    // Expects `emb` [T, NE] and `Rin` [T, HCD] and `d_pos` (*d_pos = base_pos) already in place.
    void run_front_kv(const Dev& cos_tab, const Dev& sin_tab, int T) {
        mtp_rms_norm_rows<<<T, 256, 0, stream>>>(emb.p, pre_norm_emb.p, en.p, NE, 1, EPS);
        q8_proj(fc_emb, en.p, e2.p, NE, NE, T);

        // --mtp-hnorm stream: per-stream RMSNorm over each of the 4 streams!
        mtp_rms_norm_rows<<<T * HC, 256, 0, stream>>>(Rin.p, pre_norm_hid.p, hn.p, NE, HC, EPS);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = std::min(8, T * HC - c0);
            q8_proj(fc_hid, hn.p + (size_t) c0 * NE, h2.p + (size_t) c0 * NE, NE, NE, nc);
        }
        mtp_add_streams_bcast<<<(T * HCD + 255) / 256, 256, 0, stream>>>(h2.p, e2.p, R_cur.p, NE, HC, T);

        fused_gr_norm_multi<<<dim3(HC, T), 256, 0, stream>>>(R_cur.p, ahc_norm.p, nullptr, nullptr, xn.p, NE, HC, EPS, xq32, xs32);
        launch_gr_down_inj_q8(xn.p, xq32, xs32, ahc_down.p, ahc_down.s, ahc_inj.p, lo.p, inj.p, HCD, HCLR, HC, T, stream);
        quant_act_multi<<<((HCLR * T / 16) + 127) / 128, 128, 0, stream>>>(lo.p, xq32, xs32, nullptr, nullptr, HCLR, T);
        launch_gr_up_mean_q8(xn.p, xq32, xs32, ahc_up.p, ahc_up.s, mixed.p, NE, HCLR, HC, T, stream);

        quant_act_multi<<<((NE * T / 16) + 127) / 128, 128, 0, stream>>>(mixed.p, xq32, xs32, nullptr, nullptr, NE, T);
        launch_gemm_q8p(k_proj.p, k_proj.s, xq32, xs32, kk.p, KVD, NE, T, stream);
        launch_gemm_q8p(v_proj.p, v_proj.s, xq32, xs32, vv.p, KVD, NE, T, stream);

        fused_head_norm_rope_multi<<<dim3(KVH, T), 128, 0, stream>>>(
            kk.p, k_norm.p, cos_tab.p, sin_tab.p, d_pos, KVH, HD, HD, KVD, EPS);
        kv_append_int8_multi<<<dim3(8, T), 64, 0, stream>>>(
            kk.p, vv.p, vk_q, vk_s, vv_q, vv_s, nullptr, nullptr, nullptr, nullptr, d_pos, MTP_KV_RING, MAX_CTX_256K);
    }

    // Rest half on row 0 (`R_cur[0]`, `mixed[0]`, `inj[0]`, `*d_pos` = cell of row 0):
    // Produces next draft token in `*h_out_tok`, probability in `*h_out_prob`, and leaves `R_cur[0]` copied into `Rin[0]`!
    void run_rest_row0(const Dev& cos_tab, const Dev& sin_tab) {
        q8_proj(q_proj, mixed.p, qg.p, 2 * QD, NE, 1);
        fused_head_norm_rope_multi<<<dim3(QH, 1), 128, 0, stream>>>(
            qg.p, q_norm.p, cos_tab.p, sin_tab.p, d_pos, QH, HD, 2 * HD, 2 * QD, EPS);
        mtp_qsa_attend_ring<<<QH, 256, 0, stream>>>(qg.p, vk_q, vk_s, vv_q, vv_s, d_pos, d_kv_floor, attn.p);
        q8_proj(o_proj, attn.p, bo.p, NE, QD, 1);

        // MLP hyper-connection (folds attention write into R_cur[0])
        fused_gr_norm_multi<<<dim3(HC, 1), 256, 0, stream>>>(R_cur.p, mhc_norm.p, bo.p, inj.p, xn.p, NE, HC, EPS, xq32, xs32);
        launch_gr_down_inj_q8(xn.p, xq32, xs32, mhc_down.p, mhc_down.s, mhc_inj.p, lo.p, inj2.p, HCD, HCLR, HC, 1, stream);
        quant_act_multi<<<((HCLR / 16) + 127) / 128, 128, 0, stream>>>(lo.p, xq32, xs32, nullptr, nullptr, HCLR, 1);
        launch_gr_up_mean_q8(xn.p, xq32, xs32, mhc_up.p, mhc_up.s, mixed.p, NE, HCLR, HC, 1, stream);

        quant_act_multi<<<((NE / 16) + 127) / 128, 128, 0, stream>>>(mixed.p, xq32, xs32, nullptr, nullptr, NE, 1);
        launch_gemm_bf16(router_w.p, mixed.p, slog.p, NEXP, NE, 1, stream);
        router_top10_fast_multi<<<1, 256, 0, stream>>>(slog.p, (int*) sids.p, swts.p);

        // Shared expert
        launch_gemm_q8p(sh_gate.p, sh_gate.s, xq32, xs32, sff.p, NFF, NE, 1, stream);
        launch_gemm_q8p(sh_up.p, sh_up.s, xq32, xs32, sup.p, NFF, NE, 1, stream);
        // Routed Q2_0 experts (read `xq32` before `sh_down` overwrites `xq32`!)
        mtp_exp_gu_quant_q2<<<(TOPK * (NFF / 32)), 512, 0, stream>>>(d_experts, (const int*) sids.p, TOPK, xq32, xs32, dq32, ds32);
        mtp_exp_down_2r_q2<<<(TOPK * (NE / 2) + 7) / 8, 256, 0, stream>>>(d_experts, (const int*) sids.p, TOPK, dq32, ds32, parts.p);

        silu_mul_quant32_multi<<<1, 32, 0, stream>>>(sff.p, sup.p, xq32, xs32, NFF / 32);
        launch_gemm_q8p(sh_down.p, sh_down.s, xq32, xs32, sshared.p, NE, NFF, 1, stream);
        launch_gemm_bf16(sh_gate_inp.p, mixed.p, sgate.p, 1, NE, 1, stream);

        moe_combine_shared_multi<<<dim3((NE + 255) / 256, 1), 256, 0, stream>>>(
            parts.p, swts.p, sshared.p, sgate.p, comb.p, NE, TOPK);
        gr_write_multi<<<dim3(80, 1), 256, 0, stream>>>(R_cur.p, comb.p, inj2.p, NE, HC);

        // Save `R_cur[0]` into `Rin[0]` for the next chained draft step!
        CHECK(cudaMemcpyAsync(Rin.p, R_cur.p, (size_t) HCD * 4, cudaMemcpyDeviceToDevice, stream));

        // Final mixer + 114,452 CJK subset draft head!
        fused_gr_norm_multi<<<dim3(HC, 1), 256, 0, stream>>>(R_cur.p, mix_norm.p, nullptr, nullptr, xn.p, NE, HC, EPS, xq32, xs32);
        launch_gr_down_inj_q8(xn.p, xq32, xs32, mix_down.p, mix_down.s, nullptr, lo.p, nullptr, HCD, HCLR, HC, 1, stream);
        quant_act_multi<<<((HCLR / 16) + 127) / 128, 128, 0, stream>>>(lo.p, xq32, xs32, nullptr, nullptr, HCLR, 1);
        launch_gr_up_mean_q8(xn.p, xq32, xs32, mix_up.p, mix_up.s, sample.p, NE, HCLR, HC, 1, stream);

        q8_proj(d_head, sample.p, head_logits.p, n_dvocab, NE, 1);
        mtp_head_reduce_online_part1<<<128, 256, 0, stream>>>(head_logits.p, n_dvocab, d_blk_max, d_blk_idx, d_blk_sum);
        mtp_head_finalize<<<1, 128, 0, stream>>>(d_blk_max, d_blk_idx, d_blk_sum, 128, d_dvocab, d_out_tok, d_out_prob);
        CHECK(cudaMemcpyAsync(h_out_tok, d_out_tok, 4, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(h_out_prob, d_out_prob, 4, cudaMemcpyDeviceToHost, stream));
    }

    void capture_step1(const Dev& cos_tab, const Dev& sin_tab) {
        CHECK(cudaSetDevice(dev));
        *h_pos = 0;
        CHECK(cudaMemcpyAsync(d_pos, h_pos, 4, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(emb.p, h_emb_stage, NE * 4, cudaMemcpyHostToDevice, stream));
        run_front_kv(cos_tab, sin_tab, 1);
        run_rest_row0(cos_tab, sin_tab);
        CHECK(cudaStreamSynchronize(stream));
        reset_kv();

        cudaGraph_t g = nullptr;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        CHECK(cudaMemcpyAsync(d_pos, h_pos, 4, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(emb.p, h_emb_stage, NE * 4, cudaMemcpyHostToDevice, stream));
        run_front_kv(cos_tab, sin_tab, 1);
        run_rest_row0(cos_tab, sin_tab);
        CHECK(cudaStreamEndCapture(stream, &g));
        CHECK(cudaGraphInstantiate(&step1_exec, g, nullptr, nullptr, 0));
    }
};
