// wzy_kernels_v2.cuh -- Phase 1 & Multi-Row (T=1..8) Fused/Vectorized SM75 Kernels (Zero FP64)
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

static constexpr int NE = 2560, HC = 4, HCLR = 320, HCD = HC * NE;
static constexpr int QH = 24, KVH = 2, HD = 256, QD = QH * HD, KVD = KVH * HD;
static constexpr int GH = 16, VH = 48, ST = 128, GQ = GH * ST, GV = VH * ST, QKV = 2 * GQ + GV;
static constexpr int NFF = 640, NEXP = 512, TOPK = 10, NVOCAB = 248320;
static constexpr int NROT = 64, IDX_DIM = 128, IDX_HEADS = 4, IDX_BLK = 4, IDX_TOPK = 2048;
static constexpr int MAX_W = 8; // max verify window tokens in decode / MTP (1 + 7 drafts)
static constexpr float ROPE_THETA = 1e7f, EPS = 1e-6f;
static constexpr size_t PER = 2764816;              // one NVFP4 expert blob (2.76 MB)
static constexpr int GU_ROW = NE / 64 * 36, D_ROW = NFF / 64 * 36;
static constexpr int PLE_HEAD = 160, PLE_HEADS = 16, PLE_NG = 3, PLE_HPN = 8, PLE_KERN = 4;
static constexpr int PLE_HIST = (PLE_KERN - 1) * PLE_NG;      // 9
static constexpr int PLE_EOS = 248044;

__device__ __forceinline__ float dh2f(unsigned short h) { return __half2float(*(__half*) &h); }
__device__ __forceinline__ unsigned short df2h(float f) { __half h = __float2half_rn(f); return *(unsigned short*) &h; }
__device__ __forceinline__ float db2f(unsigned short h) { return __uint_as_float((unsigned int) h << 16); }

// Gather T token embeddings from a BF16 table into FP32 rows. The table may be a device table or a UVA
// (mapped host) table, which lets the draft chain and the first stage build rows without a host roundtrip.
__global__ void mtp_gather_bf16_emb(const uint16_t* __restrict__ table, const int* __restrict__ toks,
                                    float* __restrict__ out, int T) {
    const int t = blockIdx.y;
    const int id = max(0, min(NVOCAB - 1, toks[t]));
    const uint16_t* row = table + (size_t) id * NE;
    float* dst = out + (size_t) t * NE;
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < NE; d += blockDim.x * gridDim.x) {
        dst[d] = db2f(row[d]);
    }
}

__device__ __constant__ float kUe4m3[128];
__device__ __constant__ unsigned short kPair[256];
__device__ __forceinline__ float ue4m3(unsigned x) { return kUe4m3[x & 0x7F]; }

__device__ __forceinline__ float dwsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void unpack_u32_pair(const unsigned short* tab, unsigned int w, int& lo, int& hi) {
    const int p0 = tab[w & 0xFFu];
    const int p1 = tab[(w >> 8) & 0xFFu];
    const int p2 = tab[(w >> 16) & 0xFFu];
    const int p3 = tab[w >> 24];
    lo = (p0 & 0xFF) | ((p1 & 0xFF) << 8) | ((p2 & 0xFF) << 16) | ((p3 & 0xFF) << 24);
    hi = (p0 >> 8) | ((p1 >> 8) << 8) | ((p2 >> 8) << 16) | ((p3 >> 8) << 24);
}

// Quantize T rows of width `cols` (`cols` multiple of 32) ONCE into both 32-wide (Q8_0) and 16-wide (NVFP4) int8
__global__ void quant_act_multi(const float* __restrict__ x,
                                signed char* __restrict__ q32, float* __restrict__ s32,
                                signed char* __restrict__ q16, float* __restrict__ s16,
                                int cols, int T) {
    const int total_g16 = (cols / 16) * T;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_g16) return;
    const float* xb = x + (size_t) g * 16;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 v = *(const float4*) (xb + i * 4);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    const float d = amax / 127.0f;
    const float inv_d = d > 0.f ? (127.0f / amax) : 0.f;
    if (s16) s16[g] = d;
    if (q16) {
        int4 out16;
        signed char* dst = (signed char*) &out16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int v = __float2int_rn(xb[i] * inv_d);
            dst[i] = (signed char) max(-127, min(127, v));
        }
        *(int4*) (q16 + (size_t) g * 16) = out16;
    }
    if (q32 && (g & 1) == 0) {
        float am2 = amax;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = *(const float4*) (xb + 16 + i * 4);
            am2 = fmaxf(am2, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
        const float d2 = am2 / 127.0f;
        const float inv_d2 = d2 > 0.f ? (127.0f / am2) : 0.f;
        s32[g >> 1] = d2;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            int4 out32;
            signed char* dst = (signed char*) &out32;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int v = __float2int_rn(xb[half * 16 + i] * inv_d2);
                dst[i] = (signed char) max(-127, min(127, v));
            }
            *(int4*) (q32 + (size_t) (g >> 1) * 32 + half * 16) = out32;
        }
    }
}

// Speculative window B abort: a device global (L2 resident, ~200 ns) so a wasted speculative window
// drains in microseconds instead of a full forward. It must NOT live in mapped host memory, where every
// warp would issue a PCIe read.
__device__ int g_wzy_abort = 0;

__global__ void set_wzy_abort_kernel(int v) { g_wzy_abort = v; }

static inline void set_wzy_abort(cudaStream_t s, int v) { set_wzy_abort_kernel<<<1, 1, 0, s>>>(v); }

// Multi-row Q8_0 dp4a GEMM (T = 1..8): reads the Q8_0 weight matrix ONCE across all T tokens!
template <int T>
__global__ void gemm_q8p_t(const uint8_t* __restrict__ codes, const unsigned short* __restrict__ scales,
                           const signed char* __restrict__ xq, const float* __restrict__ xs,
                           float* __restrict__ y, int rows, int cols, int y_stride) {
    if (g_wzy_abort != 0) return;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const int nb = cols / 32;
    const uint8_t* c = codes + (size_t) warp * cols;
    const unsigned short* sc = scales + (size_t) warp * nb;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;

    for (int b = lane; b < nb; b += 32) {
        const int4 a = *(const int4*) (c + (size_t) b * 32);
        const int4 bq = *(const int4*) (c + (size_t) b * 32 + 16);
        const float dw = dh2f(sc[b]);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int4 x0 = *(const int4*) (xq + (size_t) t * cols + (size_t) b * 32);
            const int4 x1 = *(const int4*) (xq + (size_t) t * cols + (size_t) b * 32 + 16);
            int isum = 0;
            isum = __dp4a(a.x, x0.x, isum);
            isum = __dp4a(a.y, x0.y, isum);
            isum = __dp4a(a.z, x0.z, isum);
            isum = __dp4a(a.w, x0.w, isum);
            isum = __dp4a(bq.x, x1.x, isum);
            isum = __dp4a(bq.y, x1.y, isum);
            isum = __dp4a(bq.z, x1.z, isum);
            isum = __dp4a(bq.w, x1.w, isum);
            acc[t] += dw * xs[(size_t) t * nb + b] * (float) isum;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const float s = dwsum(acc[t]);
        if (lane == 0) y[(size_t) t * y_stride + warp] = s;
    }
}

// Split-K variant: `S` warps cooperate on ONE output row (each takes a strided slice of the K blocks),
// then a shared-memory reduction adds the partials. The single-warp version above walks K serially with
// only ~2-3 iterations per lane, which under-hides memory latency on skinny decode GEMVs.
template <int T, int S>
__global__ void gemm_q8p_sk(const uint8_t* __restrict__ codes, const unsigned short* __restrict__ scales,
                            const signed char* __restrict__ xq, const float* __restrict__ xs,
                            float* __restrict__ y, int rows, int cols, int y_stride) {
    constexpr int WARPS = 8;
    constexpr int R = WARPS / S;                 // output rows per CTA
    __shared__ float sm[WARPS][T];
    const int wid = threadIdx.x >> 5;            // warp index INSIDE the CTA (indexes `sm`)
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * R + wid / S;
    const int sp = wid % S;
    const int nb = cols / 32;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;
    if (row < rows) {
        const uint8_t* c = codes + (size_t) row * cols;
        const unsigned short* sc = scales + (size_t) row * nb;
        for (int b = lane + 32 * sp; b < nb; b += 32 * S) {
            const int4 a  = *(const int4*) (c + (size_t) b * 32);
            const int4 bq = *(const int4*) (c + (size_t) b * 32 + 16);
            const float dw = dh2f(sc[b]);
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 x0 = *(const int4*) (xq + (size_t) t * cols + (size_t) b * 32);
                const int4 x1 = *(const int4*) (xq + (size_t) t * cols + (size_t) b * 32 + 16);
                int isum = 0;
                isum = __dp4a(a.x, x0.x, isum);
                isum = __dp4a(a.y, x0.y, isum);
                isum = __dp4a(a.z, x0.z, isum);
                isum = __dp4a(a.w, x0.w, isum);
                isum = __dp4a(bq.x, x1.x, isum);
                isum = __dp4a(bq.y, x1.y, isum);
                isum = __dp4a(bq.z, x1.z, isum);
                isum = __dp4a(bq.w, x1.w, isum);
                acc[t] += dw * xs[(size_t) t * nb + b] * (float) isum;
            }
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = dwsum(acc[t]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) sm[wid][t] = acc[t];
    }
    __syncthreads();
    if (sp == 0 && lane == 0 && row < rows) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            float tot = 0.f;
            #pragma unroll
            for (int k = 0; k < S; ++k) tot += sm[wid + k][t];
            y[(size_t) t * y_stride + row] = tot;
        }
    }
}

template <int S, int T>
static inline void launch_gemm_q8p_sk(const uint8_t* codes, const unsigned short* scales,
                                      const signed char* xq, const float* xs,
                                      float* y, int rows, int cols, int y_stride, cudaStream_t stream) {
    constexpr int R = 8 / S;
    const int blocks = (rows + R - 1) / R;
    gemm_q8p_sk<T, S><<<blocks, 256, 0, stream>>>(codes, scales, xq, xs, y, rows, cols, y_stride);
}

static inline void launch_gemm_q8p(const uint8_t* codes, const unsigned short* scales,
                                   const signed char* xq, const float* xs,
                                   float* y, int rows, int cols, int T, cudaStream_t stream, int y_stride = 0) {
    if (y_stride == 0) y_stride = rows;
    // One warp per output row: 8 warps/CTA is the historical shape. Decode GEMMs are latency-bound rather
    // than bandwidth-bound here, so the CTA width is a tuning knob (WZY_GEMM_WARPS=8|16|32).
    // WZY_GK_SPLIT=2|4: S warps cooperate on one row (split-K + shared-memory reduction). 1 = original.
    static const int gk_split = std::getenv("WZY_GK_SPLIT") ? std::atoi(std::getenv("WZY_GK_SPLIT")) : 1;
    if (gk_split == 2 || gk_split == 4) {
        int off = 0;
        while (T - off >= 8) {
            if (gk_split == 2) launch_gemm_q8p_sk<2, 8>(codes, scales, xq + (size_t) off * cols,
                                                        xs + (size_t) off * (cols / 32), y + (size_t) off * y_stride,
                                                        rows, cols, y_stride, stream);
            else               launch_gemm_q8p_sk<4, 8>(codes, scales, xq + (size_t) off * cols,
                                                        xs + (size_t) off * (cols / 32), y + (size_t) off * y_stride,
                                                        rows, cols, y_stride, stream);
            off += 8;
        }
        const int rem = T - off;
        const signed char* xqp = xq + (size_t) off * cols;
        const float* xsp = xs + (size_t) off * (cols / 32);
        float* yp = y + (size_t) off * y_stride;
        #define WZY_SK(S_, TT_) launch_gemm_q8p_sk<S_, TT_>(codes, scales, xqp, xsp, yp, rows, cols, y_stride, stream)
        if (gk_split == 2) {
            switch (rem) {
                case 1: WZY_SK(2, 1); break;
                case 2: WZY_SK(2, 2); break;
                case 3: WZY_SK(2, 3); break;
                case 4: WZY_SK(2, 4); break;
                case 5: WZY_SK(2, 5); break;
                case 6: WZY_SK(2, 6); break;
                case 7: WZY_SK(2, 7); break;
                default: break;
            }
        } else {
            switch (rem) {
                case 1: WZY_SK(4, 1); break;
                case 2: WZY_SK(4, 2); break;
                case 3: WZY_SK(4, 3); break;
                case 4: WZY_SK(4, 4); break;
                case 5: WZY_SK(4, 5); break;
                case 6: WZY_SK(4, 6); break;
                case 7: WZY_SK(4, 7); break;
                default: break;
            }
        }
        #undef WZY_SK
        return;
    }
    static const int gemm_warps = std::getenv("WZY_GEMM_WARPS") ? std::max(4, std::atoi(std::getenv("WZY_GEMM_WARPS"))) : 8;
    const int threads = gemm_warps * 32;
    const int blocks = (rows + gemm_warps - 1) / gemm_warps;
    int off = 0;
    while (T - off >= 8) {
        gemm_q8p_t<8><<<blocks, threads, 0, stream>>>(
            codes, scales, xq + (size_t) off * cols, xs + (size_t) off * (cols / 32), y + (size_t) off * y_stride, rows, cols, y_stride);
        off += 8;
    }
    const int rem = T - off;
    if (rem <= 0) return;
    const signed char* xqp = xq + (size_t) off * cols;
    const float* xsp = xs + (size_t) off * (cols / 32);
    float* yp = y + (size_t) off * y_stride;
    switch (rem) {
        case 1: gemm_q8p_t<1><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
        case 2: gemm_q8p_t<2><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
        case 3: gemm_q8p_t<3><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
        case 4: gemm_q8p_t<4><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
        case 5: gemm_q8p_t<5><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
        case 6: gemm_q8p_t<6><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
        case 7: gemm_q8p_t<7><<<blocks, threads, 0, stream>>>(codes, scales, xqp, xsp, yp, rows, cols, y_stride); break;
    }
}

// Vectorized uint4 BF16 multi-row GEMV (reads 8 BF16 weights per uint4 load, shared across T rows)
template <int T>
__global__ void gemm_bf16_t(const unsigned short* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int rows, int cols, int y_stride) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const uint4* w4 = (const uint4*) (W + (size_t) warp * cols);
    const int n8 = cols / 8;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;

    for (int j = lane; j < n8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xa = *(const float4*) (x + (size_t) t * cols + j * 8);
            const float4 xb = *(const float4*) (x + (size_t) t * cols + j * 8 + 4);
            acc[t] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                    + w4f * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const float s = dwsum(acc[t]);
        if (lane == 0) y[(size_t) t * y_stride + warp] = s;
    }
}

__global__ void gemm_bf16_row1_batch_kernel(const unsigned short* __restrict__ w_row, const float* __restrict__ x,
                                            float* __restrict__ y, int cols, int T, int y_stride) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (t >= T) return;
    const uint4* w4 = (const uint4*) w_row;
    const float* xt = x + (size_t) t * cols;
    const int n8 = cols / 8;
    float acc = 0.f;
    for (int j = lane; j < n8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float4 xa = *(const float4*) (xt + j * 8);
        const float4 xb = *(const float4*) (xt + j * 8 + 4);
        acc += __uint_as_float(wv.x << 16) * xa.x + __uint_as_float(wv.x & 0xffff0000u) * xa.y
             + __uint_as_float(wv.y << 16) * xa.z + __uint_as_float(wv.y & 0xffff0000u) * xa.w
             + __uint_as_float(wv.z << 16) * xb.x + __uint_as_float(wv.z & 0xffff0000u) * xb.y
             + __uint_as_float(wv.w << 16) * xb.z + __uint_as_float(wv.w & 0xffff0000u) * xb.w;
    }
    acc = dwsum(acc);
    if (lane == 0) y[(size_t) t * y_stride] = acc;
}

static inline void launch_gemm_bf16(const unsigned short* W, const float* x, float* y,
                                    int rows, int cols, int T, cudaStream_t stream, int y_stride = 0) {
    if (y_stride == 0) y_stride = rows;
    if (rows == 1 && T > 8) {
        gemm_bf16_row1_batch_kernel<<<(T + 7) / 8, 256, 0, stream>>>(W, x, y, cols, T, y_stride);
        return;
    }
    const int blocks = (rows + 7) / 8;
    int off = 0;
    while (T - off >= 8) {
        gemm_bf16_t<8><<<blocks, 256, 0, stream>>>(W, x + (size_t) off * cols, y + (size_t) off * y_stride, rows, cols, y_stride);
        off += 8;
    }
    const int rem = T - off;
    if (rem <= 0) return;
    const float* xp = x + (size_t) off * cols;
    float* yp = y + (size_t) off * y_stride;
    switch (rem) {
        case 1: gemm_bf16_t<1><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
        case 2: gemm_bf16_t<2><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
        case 3: gemm_bf16_t<3><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
        case 4: gemm_bf16_t<4><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
        case 5: gemm_bf16_t<5><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
        case 6: gemm_bf16_t<6><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
        case 7: gemm_bf16_t<7><<<blocks, 256, 0, stream>>>(W, xp, yp, rows, cols, y_stride); break;
    }
}

// ==================================================================
// FUSED HYPER-CONNECTION (collapses 8 graph nodes -> 3, eliminates gg/gated global traffic & cudaMemsetAsync)
// ==================================================================
// Kernel 1: optional gr_write of previous half into R + per-stream RMSNorm -> xn [T, HCD]
__global__ void quant_bf16_to_q8_kernel(const unsigned short* __restrict__ src, uint8_t* __restrict__ codes,
                                        unsigned short* __restrict__ scales, size_t n_blocks32) {
    const size_t blk = (size_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= n_blocks32) return;
    const int lane = threadIdx.x & 31;
    const float v = db2f(src[blk * 32 + lane]);
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const unsigned short sh = df2h(amax / 127.0f);
    const float d = dh2f(sh);
    const int q = d > 0.f ? max(-127, min(127, __float2int_rn(v / d))) : 0;
    codes[blk * 32 + lane] = (uint8_t) (signed char) q;
    if (lane == 0) scales[blk] = sh;
}

__global__ void fused_gr_norm_multi(float* __restrict__ R, const float* __restrict__ w_norm,
                                    const float* __restrict__ bo_prev, const float* __restrict__ inj_prev,
                                    float* __restrict__ xn, int n_embd, int hc, float eps,
                                    signed char* __restrict__ xq8 = nullptr, float* __restrict__ xs8 = nullptr) {
    const int c = blockIdx.x;
    const int t = blockIdx.y;
    const size_t base_off = ((size_t) t * hc + c) * n_embd;
    float* rc = R + base_off;
    const float* gc = w_norm + (size_t) c * n_embd;
    float* xnc = xn + base_off;
    const float gw = bo_prev ? (2.0f / (1.0f + __expf(-inj_prev[t * hc + c] / (float) hc))) : 0.0f;
    const float* bop = bo_prev ? (bo_prev + (size_t) t * n_embd) : nullptr;

    float ss = 0.f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float v = rc[d];
        if (bop) { v += bop[d] * gw; rc[d] = v; }
        ss += v * v;
    }
    __shared__ float sh[8];
    ss = dwsum(ss);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 8) ss = sh[threadIdx.x]; else ss = 0.f;
    if ((threadIdx.x >> 5) == 0) {
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (threadIdx.x == 0) sh[0] = rsqrtf(ss / (float) n_embd + eps);
    }
    __syncthreads();
    const float sc = sh[0];
    const int lane = threadIdx.x & 31;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        const float yn = rc[d] * sc * gc[d];
        xnc[d] = yn;
        if (xq8) {
            float amax = fabsf(yn);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float dq = amax / 127.0f;
            const float inv_d = dq > 0.f ? (127.0f / amax) : 0.f;
            xq8[base_off + d] = (signed char) max(-127, min(127, __float2int_rn(yn * inv_d)));
            if (lane == 0) xs8[(base_off + d) >> 5] = dq;
        }
    }
}

// Kernel 2: templated 8-rows-per-block (1 warp per row) fused hc_down + hc_inj!
// Cuts `xn` L1/L2 traffic by 8x (41 blocks instead of 321) and eliminates shared memory & __syncthreads!
template <int T>
__global__ void fused_gr_down_inj_t(const float* __restrict__ xn,
                                    const unsigned short* __restrict__ w_down,
                                    const unsigned short* __restrict__ w_inj,
                                    float* __restrict__ lo, float* __restrict__ inj,
                                    int hc_dim, int hc_lr, int hc) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n8 = hc_dim / 8;
    const int down_blocks = (hc_lr + 7) / 8; // 40

    if ((int) blockIdx.x == down_blocks) {
        if (w_inj == nullptr || warp >= hc) return;
        const uint4* row4 = (const uint4*) (w_inj + (size_t) warp * hc_dim);
        float acc[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = 0.f;
        for (int j = lane; j < n8; j += 32) {
            const uint4 wv = __ldg(row4 + j);
            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
            const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const float* xnt = xn + (size_t) t * hc_dim + j * 8;
                const float4 xa = *(const float4*) xnt;
                const float4 xb = *(const float4*) (xnt + 4);
                acc[t] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                        + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
            }
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float s = dwsum(acc[t]);
            if (lane == 0) inj[t * hc + warp] = s;
        }
        return;
    }

    const int k = blockIdx.x * 8 + warp;
    if (k >= hc_lr) return;
    const uint4* row4 = (const uint4*) (w_down + (size_t) k * hc_dim);
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;
    for (int j = lane; j < n8; j += 32) {
        const uint4 wv = __ldg(row4 + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float* xnt = xn + (size_t) t * hc_dim + j * 8;
            const float4 xa = *(const float4*) xnt;
            const float4 xb = *(const float4*) (xnt + 4);
            acc[t] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                    + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = dwsum(acc[t]) / (float) hc;
        if (lane == 0) lo[t * hc_lr + k] = v / (1.0f + __expf(-v));
    }
}

static inline void launch_fused_gr_down_inj(const float* xn, const unsigned short* w_down, const unsigned short* w_inj,
                                            float* lo, float* inj, int hc_dim, int hc_lr, int hc, int T, cudaStream_t s) {
    const int blocks = (hc_lr + 7) / 8 + (w_inj ? 1 : 0);
    int off = 0;
    while (T - off >= 8) {
        fused_gr_down_inj_t<8><<<blocks, 256, 0, s>>>(
            xn + (size_t) off * hc_dim, w_down, w_inj, lo + (size_t) off * hc_lr, inj ? (inj + off * hc) : nullptr, hc_dim, hc_lr, hc);
        off += 8;
    }
    const int rem = T - off;
    if (rem <= 0) return;
    const float* xnp = xn + (size_t) off * hc_dim;
    float* lop = lo + (size_t) off * hc_lr;
    float* injp = inj ? (inj + off * hc) : nullptr;
    switch (rem) {
        case 1: fused_gr_down_inj_t<1><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 2: fused_gr_down_inj_t<2><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 3: fused_gr_down_inj_t<3><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 4: fused_gr_down_inj_t<4><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 5: fused_gr_down_inj_t<5><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 6: fused_gr_down_inj_t<6><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 7: fused_gr_down_inj_t<7><<<blocks, 256, 0, s>>>(xnp, w_down, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
    }
}

// Kernel 3: quarter-warp stream-parallel fused hc_up + sigmoid gate + 4-stream mean!
// Each 8-lane quarter-warp (`c = lane >> 3` in 0..3, `qlane = lane & 7` in 0..7) computes stream `c` in parallel
// over `40 = 5 * 8` uint4 chunks (100% active lanes, 4x less `lo` traffic via broadcast across the 4 quarter-warps!)
template <int T>
__global__ void fused_gr_up_mean_t(const float* __restrict__ xn, const float* __restrict__ lo,
                                   const unsigned short* __restrict__ w_up,
                                   float* __restrict__ mixed, int n_embd, int hc_lr, int hc) {
    const int d = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (d >= n_embd) return;
    const int c = lane >> 3;     // 0..3 (which of the 4 streams!)
    const int qlane = lane & 7;  // 0..7 (8 lanes cover all 40 uint4s in 5 exact iterations!)
    const int row = c * n_embd + d;
    const uint4* w4p = (const uint4*) (w_up + (size_t) row * hc_lr);

    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int j = it * 8 + qlane; // 0..39
        const uint4 wv = __ldg(w4p + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float* lot = lo + (size_t) t * hc_lr + j * 8;
            const float4 xa = *(const float4*) lot;
            const float4 xb = *(const float4*) (lot + 4);
            acc[t] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                    + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
    }

    #pragma unroll
    for (int t = 0; t < T; ++t) {
        float s = acc[t];
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        float gated = 0.f;
        if (qlane == 0) {
            const float xval = xn[((size_t) t * hc + c) * n_embd + d];
            gated = xval / (1.0f + __expf(-s));
        }
        gated += __shfl_xor_sync(0xffffffffu, gated, 16);
        gated += __shfl_xor_sync(0xffffffffu, gated, 8);
        if (lane == 0) mixed[(size_t) t * n_embd + d] = gated * 0.25f;
    }
}

static inline void launch_fused_gr_up_mean(const float* xn, const float* lo, const unsigned short* w_up,
                                           float* mixed, int n_embd, int hc_lr, int hc, int T, cudaStream_t s) {
    const int blocks = (n_embd + 7) / 8;
    int off = 0;
    while (T - off >= 8) {
        fused_gr_up_mean_t<8><<<blocks, 256, 0, s>>>(
            xn + (size_t) off * hc * n_embd, lo + (size_t) off * hc_lr, w_up, mixed + (size_t) off * n_embd, n_embd, hc_lr, hc);
        off += 8;
    }
    const int rem = T - off;
    if (rem <= 0) return;
    const float* xnp = xn + (size_t) off * hc * n_embd;
    const float* lop = lo + (size_t) off * hc_lr;
    float* mxp = mixed + (size_t) off * n_embd;
    switch (rem) {
        case 1: fused_gr_up_mean_t<1><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
        case 2: fused_gr_up_mean_t<2><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
        case 3: fused_gr_up_mean_t<3><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
        case 4: fused_gr_up_mean_t<4><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
        case 5: fused_gr_up_mean_t<5><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
        case 6: fused_gr_up_mean_t<6><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
        case 7: fused_gr_up_mean_t<7><<<blocks, 256, 0, s>>>(xnp, lop, w_up, mxp, n_embd, hc_lr, hc); break;
    }
}

// Q8_0 + __dp4a Hyper-Connection Down + BF16 Inject (cuts weight bytes 2x & uses 8-way __dp4a!)
template <int T>
__global__ void gr_down_inj_q8_t(const float* __restrict__ xn,
                                 const signed char* __restrict__ xq8, const float* __restrict__ xs8,
                                 const uint8_t* __restrict__ d_codes, const unsigned short* __restrict__ d_scales,
                                 const unsigned short* __restrict__ w_inj,
                                 float* __restrict__ lo, float* __restrict__ inj,
                                 int hc_dim, int hc_lr, int hc) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int down_blocks = (hc_lr + 7) / 8; // 40

    if ((int) blockIdx.x == down_blocks) {
        if (w_inj == nullptr || warp >= hc) return;
        const uint4* row4 = (const uint4*) (w_inj + (size_t) warp * hc_dim);
        const int n8 = hc_dim / 8;
        float acc[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = 0.f;
        for (int j = lane; j < n8; j += 32) {
            const uint4 wv = __ldg(row4 + j);
            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
            const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const float* xnt = xn + (size_t) t * hc_dim + j * 8;
                const float4 xa = *(const float4*) xnt;
                const float4 xb = *(const float4*) (xnt + 4);
                acc[t] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                        + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
            }
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float s = dwsum(acc[t]);
            if (lane == 0) inj[t * hc + warp] = s;
        }
        return;
    }

    const int k = blockIdx.x * 8 + warp;
    if (k >= hc_lr) return;
    const int nb = hc_dim / 32; // 320
    const uint8_t* c_ptr = d_codes + (size_t) k * hc_dim;
    const unsigned short* s_ptr = d_scales + (size_t) k * nb;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;

    for (int b = lane; b < nb; b += 32) {
        const int4 wa = *(const int4*) (c_ptr + (size_t) b * 32);
        const int4 wb = *(const int4*) (c_ptr + (size_t) b * 32 + 16);
        const float dw = dh2f(__ldg(s_ptr + b));
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int4 x0 = *(const int4*) (xq8 + (size_t) t * hc_dim + (size_t) b * 32);
            const int4 x1 = *(const int4*) (xq8 + (size_t) t * hc_dim + (size_t) b * 32 + 16);
            int isum = __dp4a(wa.x, x0.x, __dp4a(wa.y, x0.y, __dp4a(wa.z, x0.z, __dp4a(wa.w, x0.w, 0))));
            isum = __dp4a(wb.x, x1.x, __dp4a(wb.y, x1.y, __dp4a(wb.z, x1.z, __dp4a(wb.w, x1.w, isum))));
            acc[t] = fmaf(dw * xs8[(size_t) t * nb + b], (float) isum, acc[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = dwsum(acc[t]) / (float) hc;
        if (lane == 0) lo[t * hc_lr + k] = v / (1.0f + __expf(-v));
    }
}

static inline void launch_gr_down_inj_q8(const float* xn, const signed char* xq8, const float* xs8,
                                         const uint8_t* d_codes, const unsigned short* d_scales,
                                         const unsigned short* w_inj, float* lo, float* inj,
                                         int hc_dim, int hc_lr, int hc, int T, cudaStream_t s) {
    const int blocks = (hc_lr + 7) / 8 + (w_inj ? 1 : 0);
    const int nb = hc_dim / 32;
    int off = 0;
    while (T - off >= 8) {
        gr_down_inj_q8_t<8><<<blocks, 256, 0, s>>>(
            xn + (size_t) off * hc_dim, xq8 + (size_t) off * hc_dim, xs8 + (size_t) off * nb,
            d_codes, d_scales, w_inj, lo + (size_t) off * hc_lr, inj ? (inj + off * hc) : nullptr, hc_dim, hc_lr, hc);
        off += 8;
    }
    const int rem = T - off;
    if (rem <= 0) return;
    const float* xnp = xn + (size_t) off * hc_dim;
    const signed char* xqp = xq8 + (size_t) off * hc_dim;
    const float* xsp = xs8 + (size_t) off * nb;
    float* lop = lo + (size_t) off * hc_lr;
    float* injp = inj ? (inj + off * hc) : nullptr;
    switch (rem) {
        case 1: gr_down_inj_q8_t<1><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 2: gr_down_inj_q8_t<2><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 3: gr_down_inj_q8_t<3><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 4: gr_down_inj_q8_t<4><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 5: gr_down_inj_q8_t<5><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 6: gr_down_inj_q8_t<6><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
        case 7: gr_down_inj_q8_t<7><<<blocks, 256, 0, s>>>(xnp, xqp, xsp, d_codes, d_scales, w_inj, lop, injp, hc_dim, hc_lr, hc); break;
    }
}

// Q8_0 + __dp4a Quarter-Warp Stream-Parallel Hyper-Connection Up + Mean!
// `hc_lr = 320` = `40` `int2`s (`8` bytes each) -> `5` iterations of `qlane = 0..7`, 100% coalesced!
template <int T>
__global__ void gr_up_mean_q8_t(const float* __restrict__ xn,
                                const signed char* __restrict__ lo_q8, const float* __restrict__ lo_s8,
                                const uint8_t* __restrict__ u_codes, const unsigned short* __restrict__ u_scales,
                                float* __restrict__ mixed, int n_embd, int hc_lr, int hc) {
    const int d = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (d >= n_embd) return;
    const int c = lane >> 3;     // 0..3
    const int qlane = lane & 7;  // 0..7
    const int row = c * n_embd + d;
    const int nb = hc_lr / 32;   // 10
    const uint8_t* c_ptr = u_codes + (size_t) row * hc_lr;
    const unsigned short* s_ptr = u_scales + (size_t) row * nb;

    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int j = it * 8 + qlane; // 0..39 (each `j` is 8 bytes = 2 int32s; block32 index is `j >> 2`)
        const int b = j >> 2;
        const int2 w8 = *(const int2*) (c_ptr + j * 8);
        const float dw = dh2f(__ldg(s_ptr + b));
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int2 x8 = *(const int2*) (lo_q8 + (size_t) t * hc_lr + j * 8);
            const int isum = __dp4a(w8.x, x8.x, __dp4a(w8.y, x8.y, 0));
            acc[t] = fmaf(dw * lo_s8[(size_t) t * nb + b], (float) isum, acc[t]);
        }
    }

    #pragma unroll
    for (int t = 0; t < T; ++t) {
        float s = acc[t];
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        float gated = 0.f;
        if (qlane == 0) {
            const float xval = xn[((size_t) t * hc + c) * n_embd + d];
            gated = xval / (1.0f + __expf(-s));
        }
        gated += __shfl_xor_sync(0xffffffffu, gated, 16);
        gated += __shfl_xor_sync(0xffffffffu, gated, 8);
        if (lane == 0) mixed[(size_t) t * n_embd + d] = gated * 0.25f;
    }
}

static inline void launch_gr_up_mean_q8(const float* xn, const signed char* lo_q8, const float* lo_s8,
                                        const uint8_t* u_codes, const unsigned short* u_scales,
                                        float* mixed, int n_embd, int hc_lr, int hc, int T, cudaStream_t s) {
    const int blocks = (n_embd + 7) / 8;
    const int nb = hc_lr / 32;
    int off = 0;
    while (T - off >= 8) {
        gr_up_mean_q8_t<8><<<blocks, 256, 0, s>>>(
            xn + (size_t) off * hc * n_embd, lo_q8 + (size_t) off * hc_lr, lo_s8 + (size_t) off * nb,
            u_codes, u_scales, mixed + (size_t) off * n_embd, n_embd, hc_lr, hc);
        off += 8;
    }
    const int rem = T - off;
    if (rem <= 0) return;
    const float* xnp = xn + (size_t) off * hc * n_embd;
    const signed char* lqp = lo_q8 + (size_t) off * hc_lr;
    const float* lsp = lo_s8 + (size_t) off * nb;
    float* mxp = mixed + (size_t) off * n_embd;
    switch (rem) {
        case 1: gr_up_mean_q8_t<1><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
        case 2: gr_up_mean_q8_t<2><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
        case 3: gr_up_mean_q8_t<3><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
        case 4: gr_up_mean_q8_t<4><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
        case 5: gr_up_mean_q8_t<5><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
        case 6: gr_up_mean_q8_t<6><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
        case 7: gr_up_mean_q8_t<7><<<blocks, 256, 0, s>>>(xnp, lqp, lsp, u_codes, u_scales, mxp, n_embd, hc_lr, hc); break;
    }
}

__global__ void gr_write_multi(float* __restrict__ R, const float* __restrict__ blk,
                               const float* __restrict__ inj, int n_embd, int hc) {
    const int t = blockIdx.y;
    __shared__ float sw[8];
    if ((int) threadIdx.x < hc)
        sw[threadIdx.x] = 2.0f / (1.0f + __expf(-inj[t * hc + threadIdx.x] / (float) hc));
    __syncthreads();
    const int n = hc * n_embd;
    float* Rt = R + (size_t) t * n;
    const float* bt = blk + (size_t) t * n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int c = i / n_embd, d = i - c * n_embd;
        Rt[i] += bt[d] * sw[c];
    }
}

// ==================================================================
// FUSED GDN (conv+silu+L2 in 1 kernel, ab+gate in 1 kernel, step+1/sqrt(S)+RMSNorm+sigmoid in 1 kernel!)
// Supports T >= 1 consecutive steps AND optional per-step state snapshot output for MTP speculative verify!
// ==================================================================
__global__ void __launch_bounds__(128) fused_gdn_conv_l2_multi(
    float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ kW,
    float* __restrict__ h_out, int C, int qk_heads, float eps, int T,
    float* __restrict__ hist_snaps = nullptr) {
    __shared__ float part[4];
    const int grp = blockIdx.x;          // 0 .. C/128 - 1 (80 blocks)
    const int c = grp * 128 + threadIdx.x;
    float v0 = hist[c * 3 + 0], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = kW[c * 4 + 0], w1 = kW[c * 4 + 1], w2 = kW[c * 4 + 2], w3 = kW[c * 4 + 3];

    for (int t = 0; t < T; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1; v1 = v2; v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (grp < qk_heads) {
            float sq = dwsum(y * y);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            y *= rsqrtf(ss + eps);
            __syncthreads();
        }
        h_out[(size_t) t * C + c] = y;
        if (hist_snaps) {
            float* hs = hist_snaps + (size_t) t * C * 3 + c * 3;
            hs[0] = v0; hs[1] = v1; hs[2] = v2;
        }
    }
    hist[c * 3 + 0] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}

__global__ void fused_gdn_ab_multi(const float* __restrict__ mixed,
                                   const unsigned short* __restrict__ wa,
                                   const unsigned short* __restrict__ wb,
                                   const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                   float* __restrict__ decay, float* __restrict__ bt,
                                   int n_embd, int hv, int T) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= 2 * hv) return;
    const bool is_beta = (row >= hv);
    const int r = is_beta ? (row - hv) : row;
    const uint4* w4 = (const uint4*) ((is_beta ? wb : wa) + (size_t) r * n_embd);
    const int n8 = n_embd / 8;
    const float dtr = is_beta ? 0.f : dt[r];
    const float ar = is_beta ? 0.f : ssm_a[r];

    for (int t = 0; t < T; ++t) {
        const float* xt = mixed + (size_t) t * n_embd;
        float acc = 0.f;
        for (int j = lane; j < n8; j += 32) {
            const uint4 wv = __ldg(w4 + j);
            const float4 xa = *(const float4*) (xt + j * 8);
            const float4 xb = *(const float4*) (xt + j * 8 + 4);
            acc += __uint_as_float(wv.x << 16) * xa.x + __uint_as_float(wv.x & 0xffff0000u) * xa.y
                 + __uint_as_float(wv.y << 16) * xa.z + __uint_as_float(wv.y & 0xffff0000u) * xa.w
                 + __uint_as_float(wv.z << 16) * xb.x + __uint_as_float(wv.z & 0xffff0000u) * xb.y
                 + __uint_as_float(wv.w << 16) * xb.z + __uint_as_float(wv.w & 0xffff0000u) * xb.w;
        }
        acc = dwsum(acc);
        if (lane == 0) {
            if (is_beta) {
                bt[t * hv + r] = 1.0f / (1.0f + __expf(-acc));
            } else {
                const float v = acc + dtr;
                const float sp = v > 20.0f ? v : log1pf(__expf(v));
                decay[t * hv + r] = __expf(sp * ar);
            }
        }
    }
}

// Register-resident multi-step GDN recurrence + 1/sqrt(128) + RMSNorm + sigmoid(z)!
// Grid: hv (48), Block: dim3(128, 4) = 512 threads! Each thread holds 32 rows of column `col` in registers!
__global__ void __launch_bounds__(512) fused_gdn_step_norm_multi(
    float* __restrict__ state, const float* __restrict__ h_qkv,
    const float* __restrict__ decay, const float* __restrict__ bt,
    const float* __restrict__ z, const float* __restrict__ gamma, float eps,
    float* __restrict__ y_out, int hk, int hv, int T,
    float* __restrict__ state_snaps = nullptr) {
    __shared__ float sk[128], sq[128];
    __shared__ float red_kv[4][128], red_o[4][128];
    __shared__ float wsum[4];

    const int head = blockIdx.x;     // 0..47
    const int col = threadIdx.x;     // 0..127
    const int rg = threadIdx.y;      // 0..3 (each rg owns 32 rows)
    const int tid = rg * 128 + col;
    const int qh = head % hk;
    const size_t row_stride = (size_t) hv * 128;
    const int C = 2 * hk * 128 + hv * 128; // 10240
    const int GV_W = hv * 128;             // 6144

    float s[32];
    float* base = state + ((size_t) (rg * 32) * hv + head) * 128 + col;
    #pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = base[r * row_stride];
    const float gm = gamma[col];

    for (int t = 0; t < T; ++t) {
        const float* ht = h_qkv + (size_t) t * C;
        if (tid < 128) {
            sq[tid] = ht[qh * 128 + tid];
            sk[tid] = ht[hk * 128 + qh * 128 + tid];
        }
        __syncthreads();

        const float dec = decay[t * hv + head];
        const float beta = bt[t * hv + head];
        float kv = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            s[r] *= dec;
            kv = fmaf(s[r], sk[rg * 32 + r], kv);
        }
        red_kv[rg][col] = kv;
        __syncthreads();

        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * hk * 128 + head * 128 + col] - kv_col) * beta;
        float o = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            s[r] = fmaf(sk[rg * 32 + r], delta, s[r]);
            o = fmaf(s[r], sq[rg * 32 + r], o);
            if (state_snaps) {
                float* sbase = state_snaps + (size_t) t * (128 * hv * 128) + ((size_t) (rg * 32 + r) * hv + head) * 128 + col;
                *sbase = s[r];
            }
        }
        red_o[rg][col] = o;
        __syncthreads();

        if (rg == 0) {
            const float oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * 0.08838834764831845f; // rsqrt(128)
            float sq_part = dwsum(oc * oc);
            if ((col & 31) == 0) wsum[col >> 5] = sq_part;
            __syncthreads();
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss * (1.0f / 128.0f) + eps);
            const float zz = z[(size_t) t * GV_W + head * 128 + col];
            y_out[(size_t) t * GV_W + head * 128 + col] = oc * scale * gm / (1.0f + __expf(-zz));
        } else {
            __syncthreads();
        }
        if (t + 1 < T) __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 32; ++r) base[r * row_stride] = s[r];
}

// ==================================================================
// ZERO-FP64 PARALLEL ROUTER TOP-10 (replaces slow FP64 serial router_topk!)
// ==================================================================
__global__ void router_top10_fast_multi(const float* __restrict__ logits, int* __restrict__ ids,
                                        float* __restrict__ wts) {
    const int t = blockIdx.x;
    const float* x = logits + (size_t) t * NEXP;
    int* out_id = ids + (size_t) t * TOPK;
    float* out_w = wts + (size_t) t * TOPK;

    __shared__ float sp[NEXP];
    __shared__ float wr[8];
    const int tid = threadIdx.x; // 256 threads
    const int lane = tid & 31, warp = tid >> 5;

    float v0 = x[tid], v1 = x[tid + 256];
    float mx = fmaxf(v0, v1);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) wr[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = (lane < 8) ? wr[lane] : -1e30f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (lane == 0) wr[0] = mx;
    }
    __syncthreads();
    mx = wr[0];

    const float e0 = __expf(v0 - mx), e1 = __expf(v1 - mx);
    float es = dwsum(e0 + e1);
    if (lane == 0) wr[warp] = es;
    __syncthreads();
    if (warp == 0) {
        es = (lane < 8) ? wr[lane] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) es += __shfl_xor_sync(0xffffffffu, es, o);
        if (lane == 0) wr[0] = 1.0f / es;
    }
    __syncthreads();
    const float inv_s = wr[0];
    sp[tid] = e0 * inv_s;
    sp[tid + 256] = e1 * inv_s;
    __syncthreads();

    // Warp 0 performs 10 parallel warp-wide argmax passes over the 512 probabilities in shared memory!
    if (warp == 0) {
        float wsum10 = 0.f;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float bv = -1.f;
            int bi = NEXP;
            #pragma unroll
            for (int base = 0; base < NEXP; base += 32) {
                const int idx = base + lane;
                const float pv = sp[idx];
                if (pv > bv || (pv == bv && idx < bi)) { bv = pv; bi = idx; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                out_id[k] = bi;
                out_w[k] = bv;
                sp[bi] = -2.f; // mask selected
                wsum10 += bv;
            }
            __syncwarp();
        }
        if (lane == 0) {
            const float sc = fmaxf(wsum10, 6.103515625e-05f);
            #pragma unroll
            for (int k = 0; k < TOPK; ++k) out_w[k] /= sc;
        }
    }
}

// ==================================================================
// VECTORIZED uint32 NVFP4 ROUTED EXPERTS (multi-row T >= 1, 4x fewer load instructions!)
// ==================================================================
__global__ void map_slots_lut_multi(const int* __restrict__ ids, int total_k, const int* __restrict__ lut,
                                    int layer, int* __restrict__ slots, int* __restrict__ misses,
                                    int* __restrict__ miss_key) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_k) return;
    const int gkey = layer * NEXP + ids[i];
    const int s = lut[gkey];
    slots[i] = (s >= 0) ? s : -(1 + gkey);
    if (s < 0) {
        const int m = atomicAdd(misses, 1);
        if (m < 2048) miss_key[m] = gkey;
    }
}

__device__ __forceinline__ const unsigned char* blob_at_hybrid(
    const unsigned char* vram_blobs, const unsigned char* host_pack, int slot) {
    return (slot >= 0) ? (vram_blobs + (size_t) slot * PER)
                       : (host_pack + (size_t) (-1 - slot) * PER);
}

__global__ void exp_scales_multi(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                                 const int* __restrict__ slots, int total_k, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total_k) return;
    const float* tail = (const float*) (blob_at_hybrid(b0, hpack, slots[e]) + PER - 16);
    out[3 * e + 0] = tail[0];
    out[3 * e + 1] = tail[1];
    out[3 * e + 2] = tail[2];
}

// Vectorized exp_gu: total_k = T * TOPK; token index t = e / TOPK reads xq + t * NE!
__global__ void exp_gu_vec(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                           const int* __restrict__ slots, int total_k,
                           const signed char* __restrict__ xq, const float* __restrict__ xs,
                           float* __restrict__ act_out, const float* __restrict__ scales) {
    __shared__ unsigned short tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = kPair[i];
    __syncthreads();

    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int e = gwarp / NFF;
    if (e >= total_k) return;
    const int row = gwarp - e * NFF;
    const int t = e / TOPK;

    const unsigned char* base = blob_at_hybrid(b0, hpack, slots[e]);
    const unsigned char* g_ptr = base + (size_t) row * GU_ROW;
    const unsigned char* u_ptr = base + (size_t) (NFF + row) * GU_ROW;
    const signed char* xqt = xq + (size_t) t * NE;
    const float* xst = xs + (size_t) t * (NE / 16);
    const int nb = NE / 64; // 40

    float ag = 0.f, au = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int4 a0 = *(const int4*) (xqt + b * 64);
        const int4 a1 = *(const int4*) (xqt + b * 64 + 16);
        const int4 a2 = *(const int4*) (xqt + b * 64 + 32);
        const int4 a3 = *(const int4*) (xqt + b * 64 + 48);
        const float4 sf = *(const float4*) (xst + b * 4);
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const uint32_t* blk32 = (const uint32_t*) ((pass ? u_ptr : g_ptr) + b * 36);
            const uint32_t ds = blk32[0];
            float acc = 0.f;
            int lo0, hi0, lo1, hi1;

            unpack_u32_pair(tab, blk32[1], lo0, hi0);
            unpack_u32_pair(tab, blk32[2], lo1, hi1);
            acc += ue4m3(ds & 0xFFu) * sf.x * (float) (__dp4a(lo0, a0.x, 0) + __dp4a(lo1, a0.y, 0) + __dp4a(hi0, a0.z, 0) + __dp4a(hi1, a0.w, 0));

            unpack_u32_pair(tab, blk32[3], lo0, hi0);
            unpack_u32_pair(tab, blk32[4], lo1, hi1);
            acc += ue4m3((ds >> 8) & 0xFFu) * sf.y * (float) (__dp4a(lo0, a1.x, 0) + __dp4a(lo1, a1.y, 0) + __dp4a(hi0, a1.z, 0) + __dp4a(hi1, a1.w, 0));

            unpack_u32_pair(tab, blk32[5], lo0, hi0);
            unpack_u32_pair(tab, blk32[6], lo1, hi1);
            acc += ue4m3((ds >> 16) & 0xFFu) * sf.z * (float) (__dp4a(lo0, a2.x, 0) + __dp4a(lo1, a2.y, 0) + __dp4a(hi0, a2.z, 0) + __dp4a(hi1, a2.w, 0));

            unpack_u32_pair(tab, blk32[7], lo0, hi0);
            unpack_u32_pair(tab, blk32[8], lo1, hi1);
            acc += ue4m3(ds >> 24) * sf.w * (float) (__dp4a(lo0, a3.x, 0) + __dp4a(lo1, a3.y, 0) + __dp4a(hi0, a3.z, 0) + __dp4a(hi1, a3.w, 0));

            if (pass == 0) ag += acc; else au += acc;
        }
    }
    ag = dwsum(ag) * scales[3 * e + 0];
    au = dwsum(au) * scales[3 * e + 1];
    // Fused SwiGLU epilogue right inside exp_gu_vec (saves exp_gu_epi launch + ff3 global memory!)
    if (lane == 0) {
        act_out[(size_t) e * NFF + row] = (ag / (1.0f + __expf(-ag))) * au;
    }
}

__global__ void exp_down_vec(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                             const int* __restrict__ slots, int total_k,
                             const signed char* __restrict__ xq, const float* __restrict__ xs,
                             float* __restrict__ out, const float* __restrict__ scales) {
    __shared__ unsigned short tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = kPair[i];
    __syncthreads();

    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int e = gwarp / NE;
    if (e >= total_k) return;
    const int row = gwarp - e * NE;

    const unsigned char* base = blob_at_hybrid(b0, hpack, slots[e]) + 2 * (size_t) NFF * GU_ROW;
    const unsigned char* r_ptr = base + (size_t) row * D_ROW;
    const signed char* xqe = xq + (size_t) e * NFF;
    const float* xse = xs + (size_t) e * (NFF / 16);
    const int nb = NFF / 64; // 10

    float acc = 0.f;
    if (lane < nb) {
        const uint32_t* blk32 = (const uint32_t*) (r_ptr + lane * 36);
        const uint32_t ds = blk32[0];
        const int4 a0 = *(const int4*) (xqe + lane * 64);
        const int4 a1 = *(const int4*) (xqe + lane * 64 + 16);
        const int4 a2 = *(const int4*) (xqe + lane * 64 + 32);
        const int4 a3 = *(const int4*) (xqe + lane * 64 + 48);
        const float4 sf = *(const float4*) (xse + lane * 4);
        int lo0, hi0, lo1, hi1;

        unpack_u32_pair(tab, blk32[1], lo0, hi0);
        unpack_u32_pair(tab, blk32[2], lo1, hi1);
        acc += ue4m3(ds & 0xFFu) * sf.x * (float) (__dp4a(lo0, a0.x, 0) + __dp4a(lo1, a0.y, 0) + __dp4a(hi0, a0.z, 0) + __dp4a(hi1, a0.w, 0));

        unpack_u32_pair(tab, blk32[3], lo0, hi0);
        unpack_u32_pair(tab, blk32[4], lo1, hi1);
        acc += ue4m3((ds >> 8) & 0xFFu) * sf.y * (float) (__dp4a(lo0, a1.x, 0) + __dp4a(lo1, a1.y, 0) + __dp4a(hi0, a1.z, 0) + __dp4a(hi1, a1.w, 0));

        unpack_u32_pair(tab, blk32[5], lo0, hi0);
        unpack_u32_pair(tab, blk32[6], lo1, hi1);
        acc += ue4m3((ds >> 16) & 0xFFu) * sf.z * (float) (__dp4a(lo0, a2.x, 0) + __dp4a(lo1, a2.y, 0) + __dp4a(hi0, a2.z, 0) + __dp4a(hi1, a2.w, 0));

        unpack_u32_pair(tab, blk32[7], lo0, hi0);
        unpack_u32_pair(tab, blk32[8], lo1, hi1);
        acc += ue4m3(ds >> 24) * sf.w * (float) (__dp4a(lo0, a3.x, 0) + __dp4a(lo1, a3.y, 0) + __dp4a(hi0, a3.z, 0) + __dp4a(hi1, a3.w, 0));
    }
    acc = dwsum(acc) * scales[3 * e + 2];
    if (lane == 0) out[(size_t) e * NE + row] = acc;
}

__global__ void silu_mul_multi(const float* __restrict__ gate, const float* __restrict__ up,
                               float* __restrict__ out, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = (gate[i] / (1.0f + __expf(-gate[i]))) * up[i];
}

// Fused SiLU(gate)*up + 32-wide Q8_0 activation quantization (eliminates 49 kernel launches + global memory round-trips per window!)
__global__ void silu_mul_quant32_multi(const float* __restrict__ gate, const float* __restrict__ up,
                                       signed char* __restrict__ q32, float* __restrict__ s32, int total_g32) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_g32) return;
    const float* gp = gate + (size_t) g * 32;
    const float* up_p = up + (size_t) g * 32;
    float v[32];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 gv = *(const float4*) (gp + i * 4);
        const float4 uv = *(const float4*) (up_p + i * 4);
        v[i * 4 + 0] = (gv.x / (1.0f + __expf(-gv.x))) * uv.x;
        v[i * 4 + 1] = (gv.y / (1.0f + __expf(-gv.y))) * uv.y;
        v[i * 4 + 2] = (gv.z / (1.0f + __expf(-gv.z))) * uv.z;
        v[i * 4 + 3] = (gv.w / (1.0f + __expf(-gv.w))) * uv.w;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[i * 4 + 0]), fabsf(v[i * 4 + 1])),
                                 fmaxf(fabsf(v[i * 4 + 2]), fabsf(v[i * 4 + 3]))));
    }
    const float d = amax / 127.0f;
    const float inv_d = d > 0.f ? (127.0f / amax) : 0.f;
    s32[g] = d;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        int4 out16;
        signed char* dst = (signed char*) &out16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int q = __float2int_rn(v[half * 16 + i] * inv_d);
            dst[i] = (signed char) max(-127, min(127, q));
        }
        *(int4*) (q32 + (size_t) g * 32 + half * 16) = out16;
    }
}
__global__ void moe_combine_shared_multi(const float* __restrict__ parts, const float* __restrict__ wts,
                                         const float* __restrict__ shared, const float* __restrict__ sgate,
                                         float* __restrict__ out, int n, int k) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    const float sg = 1.0f / (1.0f + __expf(-sgate[t]));
    float s = shared[(size_t) t * n + d] * sg;
    const float* wt = wts + (size_t) t * k;
    const float* pt = parts + (size_t) t * k * n;
    #pragma unroll
    for (int e = 0; e < 10; ++e) { const float w = wt[e]; if (w != 0.0f) s += w * pt[(size_t) e * n + d]; }
    out[(size_t) t * n + d] = s;
}

// ==================================================================
// PLE MULTI-ROW (Layer 1, supports T >= 1 + optional state snapshot for MTP verify)
// ==================================================================
__global__ void ple_group_multi(const float* __restrict__ x, const float* __restrict__ gamma,
                                float* __restrict__ y, int n_embd, int hc, float eps) {
    const int c = blockIdx.x;
    const int t = blockIdx.y;
    const float* xc = x + ((size_t) t * hc + c) * n_embd;
    float* yc = y + ((size_t) t * hc + c) * n_embd;
    const float* gc = gamma + (size_t) c * n_embd;
    __shared__ float red[8];
    float ss = 0.f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) ss += xc[d] * xc[d];
    ss = dwsum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        red[0] = rsqrtf(tot / (float) n_embd + eps);
    }
    __syncthreads();
    const float rs = red[0];
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) yc[d] = xc[d] * rs * gc[d];
}

__global__ void ple_gate_gv_multi(const float* __restrict__ key, const float* __restrict__ query,
                                  const float* __restrict__ value, float* __restrict__ gv,
                                  int hc, int n_embd) {
    const int c = blockIdx.x;
    const int t = blockIdx.y;
    const float* kc = key + ((size_t) t * hc + c) * n_embd;
    const float* qc = query + ((size_t) t * hc + c) * n_embd;
    const float* vt = value + (size_t) t * n_embd;
    float* gvc = gv + ((size_t) t * hc + c) * n_embd;

    __shared__ float red[8];
    float acc = 0.f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) acc += kc[d] * qc[d];
    acc = dwsum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        tot /= sqrtf((float) n_embd);
        const float sg = tot < 0.f ? -1.f : 1.f;
        red[0] = 1.f / (1.f + __expf(-sg * sqrtf(fmaxf(fabsf(tot), 1e-6f))));
    }
    __syncthreads();
    const float g = red[0];
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) gvc[d] = vt[d] * g;
}

__global__ void ple_conv_multi(const float* __restrict__ norm, const float* __restrict__ cw,
                               float* __restrict__ hist, const float* __restrict__ gv,
                               float* __restrict__ R, int hcd, int ng, int kern, int T,
                               float* __restrict__ hist_snaps = nullptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hcd) return;
    float h[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) h[i] = hist[(size_t) i + (size_t) 9 * c];
    const float w0 = cw[0 + 4 * c], w1 = cw[1 + 4 * c], w2 = cw[2 + 4 * c], w3 = cw[3 + 4 * c];

    for (int t = 0; t < T; ++t) {
        const float cur = norm[(size_t) t * hcd + c];
        const float acc = w0 * h[0] + w1 * h[3] + w2 * h[6] + w3 * cur;
        #pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = h[i + 1];
        h[8] = cur;
        R[(size_t) t * hcd + c] += gv[(size_t) t * hcd + c] + acc / (1.0f + __expf(-acc));
        if (hist_snaps) {
            float* hs = hist_snaps + (size_t) t * hcd * 9 + (size_t) 9 * c;
            #pragma unroll
            for (int i = 0; i < 9; ++i) hs[i] = h[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 9; ++i) hist[(size_t) i + (size_t) 9 * c] = h[i];
}

// ==================================================================
// GROUPED MULTI-TOKEN NVFP4 EXPERTS & UNLOCKED TENSOR-CORE HELPERS
// Groups `total_k = T * TOPK` routed entries by expert ID (up to GRP_MAX = 8 tokens per group)
// so shared experts across a verify window (W=6) or prefill chunk (T=128) read DRAM ONCE!
// ==================================================================
#ifndef WZY_GRP_MAX
#define WZY_GRP_MAX 4
#endif
static constexpr int GRP_MAX = WZY_GRP_MAX;

__global__ void group_experts_kernel(const int* __restrict__ sids, const int* __restrict__ dslots,
                                     int total_k, int* __restrict__ grp_slot, int* __restrict__ grp_cnt,
                                     int* __restrict__ grp_e, int* __restrict__ n_groups) {
    const int eid = threadIdx.x; // 0..511 (one thread per possible expert!)
    if (eid == 0) *n_groups = 0;
    __syncthreads();
    if (eid >= NEXP) return;

    int buf[GRP_MAX];
    int c = 0;
    int my_slot = 0;
    for (int e = 0; e < total_k; ++e) {
        if (sids[e] == eid) {
            if (c == 0) my_slot = dslots[e];
            buf[c++] = e;
            if (c == GRP_MAX) {
                const int g = atomicAdd(n_groups, 1);
                grp_slot[g] = my_slot;
                grp_cnt[g] = c;
                #pragma unroll
                for (int j = 0; j < GRP_MAX; ++j) grp_e[g * GRP_MAX + j] = buf[j];
                c = 0;
            }
        }
    }
    if (c > 0) {
        const int g = atomicAdd(n_groups, 1);
        grp_slot[g] = my_slot;
        grp_cnt[g] = c;
        for (int j = 0; j < c; ++j) grp_e[g * GRP_MAX + j] = buf[j];
    }
}

// 1-cycle hardware `PRMT.B32` (`__byte_perm`) SIMD NVFP4 unpacker (zero 64-bit shifts, zero shared memory!)
__device__ __forceinline__ int nvfp4_pack4_prmt(unsigned int nibs_at_bytes) {
    const unsigned int sel = (nibs_at_bytes & 7u)
                           | ((nibs_at_bytes >> 4) & 0x70u)
                           | ((nibs_at_bytes >> 8) & 0x700u)
                           | ((nibs_at_bytes >> 12) & 0x7000u);
    const unsigned int mag = __byte_perm(0x03020100u, 0x0C080604u, sel);
    const unsigned int nz  = nibs_at_bytes | (nibs_at_bytes >> 1) | (nibs_at_bytes >> 2);
    const unsigned int sgn = (nibs_at_bytes >> 3) & nz & 0x01010101u;
    const unsigned int msk = sgn * 0xFFu;
    return (int) ((mag ^ msk) + sgn);
}

__device__ __forceinline__ void unpack_nvfp4_sub16_reg(unsigned int w0, unsigned int w1,
                                                       int& lo0, int& lo1, int& hi0, int& hi1) {
    lo0 = nvfp4_pack4_prmt(w0);
    hi0 = nvfp4_pack4_prmt(w0 >> 4);
    lo1 = nvfp4_pack4_prmt(w1);
    hi1 = nvfp4_pack4_prmt(w1 >> 4);
}

// 4-rows-per-warp (32 rows per 256-thread block!), 100%-lane-utilized, PRMT-accelerated `exp_gu_grouped`:
// Launch grid: `(max_groups * NFF + 31) / 32`!
// `NE = 2560` = `160 = 20 * 8` sub-blocks per row -> 4 rows share `xq` in L1 cache and 100% of lanes are active!
__global__ void exp_gu_grouped(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                               const int* __restrict__ grp_slot, const int* __restrict__ grp_cnt,
                               const int* __restrict__ grp_e, const int* __restrict__ n_groups,
                               int max_groups, const signed char* __restrict__ xq, const float* __restrict__ xs,
                               float* __restrict__ act_out, const float* __restrict__ scales) {
    (void) scales;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int is_up = lane >> 4;   // 0 = Gate (lanes 0..15), 1 = Up (lanes 16..31)
    const int hlane = lane & 15;   // 0..15 (16 lanes cover 160 = 10 * 16 sub-blocks with 100% active lanes!)
    const int g = gwarp / NFF;
    if (g >= max_groups || g >= *n_groups) return;
    const int row = gwarp - g * NFF;
    const int cnt = grp_cnt[g];

    int e_idx[GRP_MAX];
    int t_idx[GRP_MAX];
    float acc[GRP_MAX];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        const int ej = (j < cnt) ? grp_e[g * GRP_MAX + j] : 0;
        e_idx[j] = ej;
        t_idx[j] = ej / TOPK;
        acc[j] = 0.f;
    }

    const unsigned char* base = blob_at_hybrid(b0, hpack, grp_slot[g]);
    const unsigned char* r_ptr = base + (size_t) (is_up * NFF + row) * GU_ROW;

    #pragma unroll
    for (int it = 0; it < 10; ++it) {
        const int sub = it * 16 + hlane; // 0..159 (lanes 0..15 and 16..31 share the EXACT same `sub` of `xq`!)
        const int b = sub >> 2;          // 0..39
        const int s = sub & 3;           // 0..3
        const unsigned char* rb = r_ptr + b * 36;

        const float ws = ue4m3(__ldg(rb + s));
        const uint32_t* wp32 = (const uint32_t*) (rb + 4 + s * 8);
        int w0, w1, w2, w3;
        unpack_nvfp4_sub16_reg(__ldg(wp32), __ldg(wp32 + 1), w0, w1, w2, w3);

        #pragma unroll
        for (int j = 0; j < GRP_MAX; ++j) {
            if (j < cnt) {
                const int t = t_idx[j];
                const int4 av = *(const int4*) (xq + (size_t) t * NE + sub * 16);
                const float xsc = xs[(size_t) t * (NE / 16) + sub];
                const int d0 = __dp4a(w0, av.x, __dp4a(w1, av.y, __dp4a(w2, av.z, __dp4a(w3, av.w, 0))));
                acc[j] = fmaf(ws * xsc, (float) d0, acc[j]);
            }
        }
    }

    const float* tail = (const float*) (base + PER - 16);
    const float my_sc = tail[is_up];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        if (j < cnt) {
            float v = acc[j];
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v *= my_sc;
            const float usum = __shfl_sync(0xffffffffu, v, 16);
            if (lane == 0) {
                act_out[(size_t) e_idx[j] * NFF + row] = (v / (1.0f + __expf(-v))) * usum;
            }
        }
    }
}
// 4-rows-per-warp (32 rows per 256-thread block!), 100%-lane-utilized, coalesced `exp_down_grouped`:
// Launch grid: `(max_groups * NE + 31) / 32`!
// `NFF = 640` = `40 = 5 * 8` sub-blocks per row -> 8 lanes per row x 4 rows per warp = 100% active lanes!
__global__ void exp_down_grouped(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                                 const int* __restrict__ grp_slot, const int* __restrict__ grp_cnt,
                                 const int* __restrict__ grp_e, const int* __restrict__ n_groups,
                                 int max_groups, const signed char* __restrict__ xq, const float* __restrict__ xs,
                                 float* __restrict__ out, const float* __restrict__ scales) {
    (void) scales;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int qrow = lane >> 3;  // 0..3 (which of the 4 rows in this warp)
    const int qlane = lane & 7;  // 0..7 (8 lanes cooperate on the 40 sub-blocks of 1 row!)
    const int g = gwarp / (NE / 4); // NE / 4 = 640
    if (g >= max_groups || g >= *n_groups) return;
    const int row = (gwarp - g * (NE / 4)) * 4 + qrow;
    const int cnt = grp_cnt[g];

    int e_idx[GRP_MAX];
    float acc[GRP_MAX];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        e_idx[j] = (j < cnt) ? grp_e[g * GRP_MAX + j] : 0;
        acc[j] = 0.f;
    }

    const unsigned char* blob = blob_at_hybrid(b0, hpack, grp_slot[g]);
    const unsigned char* base = blob + 2 * (size_t) NFF * GU_ROW;
    const unsigned char* r_ptr = base + (size_t) row * D_ROW;

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int sub = it * 8 + qlane; // 0..39 (100% of all 8 lanes active across all 5 iterations!)
        const int b = sub >> 2;         // 0..9
        const int s = sub & 3;          // 0..3
        const unsigned char* db = r_ptr + b * 36;
        const float ds = ue4m3(__ldg(db + s));
        const uint32_t* dp32 = (const uint32_t*) (db + 4 + s * 8);

        int dw0, dw1, dw2, dw3;
        unpack_nvfp4_sub16_reg(__ldg(dp32), __ldg(dp32 + 1), dw0, dw1, dw2, dw3);

        #pragma unroll
        for (int j = 0; j < GRP_MAX; ++j) {
            if (j < cnt) {
                const int ej = e_idx[j];
                const int4 av = *(const int4*) (xq + (size_t) ej * NFF + sub * 16);
                const float xsc = xs[(size_t) ej * (NFF / 16) + sub];
                const int d0 = __dp4a(dw0, av.x, __dp4a(dw1, av.y, __dp4a(dw2, av.z, __dp4a(dw3, av.w, 0))));
                acc[j] = fmaf(ds * xsc, (float) d0, acc[j]);
            }
        }
    }

    const float sc_d = ((const float*) (blob + PER - 16))[2];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        if (j < cnt) {
            float v = acc[j];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if (qlane == 0) out[(size_t) e_idx[j] * NE + row] = v * sc_d;
        }
    }
}

// 2-rows-per-warp (16 rows per 256-thread block = 1 exact 16-wide quantization block!) + Fused SwiGLU & INT8 `dq16/ds16` epilogue!
// Launch grid: `dim3(NFF / 16, max_groups)` = `dim3(40, max_groups)`, Block: 256
__global__ void __launch_bounds__(256) exp_gu_quant_grouped(
    const unsigned char* __restrict__ b0,
    const int* __restrict__ grp_slot, const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_e, const int* __restrict__ n_groups,
    const signed char* __restrict__ xq, const float* __restrict__ xs,
    signed char* __restrict__ dq16, float* __restrict__ ds16) {

    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int blk16 = blockIdx.x; // 0..39
    const int warp = threadIdx.x >> 5; // 0..7
    const int lane = threadIdx.x & 31;
    const int is_up = lane >> 4;
    const int hlane = lane & 15;
    const int row0 = blk16 * 16 + warp * 2;
    const int cnt = grp_cnt[g];

    // `grp_e[]` holds flat routed-entry indices (`e = token * TOPK + k`), so the token is `e / TOPK`.
    // Keeping those GRP_MAX indices per thread cost 8 registers at GRP_MAX=8 (the 16B spill source);
    // one broadcast shared row replaces them for a single `__syncthreads()` and zero registers.
    __shared__ int sh_t[GRP_MAX];
    if (threadIdx.x < GRP_MAX) {
        sh_t[threadIdx.x] = (threadIdx.x < cnt) ? (grp_e[g * GRP_MAX + threadIdx.x] / TOPK) : 0;
    }
    __syncthreads();
    float acc0[GRP_MAX], acc1[GRP_MAX];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        acc0[j] = 0.f;
        acc1[j] = 0.f;
    }

    const unsigned char* base = b0 + (size_t) grp_slot[g] * PER;
    const unsigned char* r_ptr0 = base + (size_t) (is_up * NFF + row0) * GU_ROW;
    const unsigned char* r_ptr1 = r_ptr0 + GU_ROW;

    if (cnt == 1) {
        const signed char* xq0 = xq + (size_t) sh_t[0] * NE;
        const float* xs0 = xs + (size_t) sh_t[0] * (NE / 16);
        #pragma unroll
        for (int it = 0; it < 10; ++it) {
            const int sub = it * 16 + hlane;
            const int b = sub >> 2;
            const int s = sub & 3;
            const int4 av = *(const int4*) (xq0 + sub * 16);
            const float xsc = xs0[sub];

            const unsigned char* rb0 = r_ptr0 + b * 36;
            const float ws0 = ue4m3(__ldg(rb0 + s));
            const uint32_t* wp0 = (const uint32_t*) (rb0 + 4 + s * 8);
            int w0a, w1a, w2a, w3a;
            unpack_nvfp4_sub16_reg(__ldg(wp0), __ldg(wp0 + 1), w0a, w1a, w2a, w3a);

            const unsigned char* rb1 = r_ptr1 + b * 36;
            const float ws1 = ue4m3(__ldg(rb1 + s));
            const uint32_t* wp1 = (const uint32_t*) (rb1 + 4 + s * 8);
            int w0b, w1b, w2b, w3b;
            unpack_nvfp4_sub16_reg(__ldg(wp1), __ldg(wp1 + 1), w0b, w1b, w2b, w3b);

            const int d0 = __dp4a(w0a, av.x, __dp4a(w1a, av.y, __dp4a(w2a, av.z, __dp4a(w3a, av.w, 0))));
            const int d1 = __dp4a(w0b, av.x, __dp4a(w1b, av.y, __dp4a(w2b, av.z, __dp4a(w3b, av.w, 0))));
            acc0[0] = fmaf(ws0 * xsc, (float) d0, acc0[0]);
            acc1[0] = fmaf(ws1 * xsc, (float) d1, acc1[0]);
        }
    } else {
        #pragma unroll
        for (int it = 0; it < 10; ++it) {
            const int sub = it * 16 + hlane;
            const int b = sub >> 2;
            const int s = sub & 3;

            const unsigned char* rb0 = r_ptr0 + b * 36;
            const float ws0 = ue4m3(__ldg(rb0 + s));
            const uint32_t* wp0 = (const uint32_t*) (rb0 + 4 + s * 8);
            int w0a, w1a, w2a, w3a;
            unpack_nvfp4_sub16_reg(__ldg(wp0), __ldg(wp0 + 1), w0a, w1a, w2a, w3a);

            const unsigned char* rb1 = r_ptr1 + b * 36;
            const float ws1 = ue4m3(__ldg(rb1 + s));
            const uint32_t* wp1 = (const uint32_t*) (rb1 + 4 + s * 8);
            int w0b, w1b, w2b, w3b;
            unpack_nvfp4_sub16_reg(__ldg(wp1), __ldg(wp1 + 1), w0b, w1b, w2b, w3b);

            #pragma unroll
            for (int j = 0; j < GRP_MAX; ++j) {
                if (j < cnt) {
                    const int t = sh_t[j];
                    const int4 av = *(const int4*) (xq + (size_t) t * NE + sub * 16);
                    const float xsc = xs[(size_t) t * (NE / 16) + sub];
                    const int d0 = __dp4a(w0a, av.x, __dp4a(w1a, av.y, __dp4a(w2a, av.z, __dp4a(w3a, av.w, 0))));
                    const int d1 = __dp4a(w0b, av.x, __dp4a(w1b, av.y, __dp4a(w2b, av.z, __dp4a(w3b, av.w, 0))));
                    acc0[j] = fmaf(ws0 * xsc, (float) d0, acc0[j]);
                    acc1[j] = fmaf(ws1 * xsc, (float) d1, acc1[j]);
                }
            }
        }
    }

    __shared__ float sh_y[GRP_MAX][16];
    const float my_sc = ((const float*) (base + PER - 16))[is_up];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        if (j < cnt) {
            float v0 = acc0[j], v1 = acc1[j];
            v0 += __shfl_xor_sync(0xffffffffu, v0, 8);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 8);
            v0 += __shfl_xor_sync(0xffffffffu, v0, 4);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 4);
            v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
            v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
            v0 *= my_sc;
            v1 *= my_sc;
            const float u0 = __shfl_sync(0xffffffffu, v0, 16);
            const float u1 = __shfl_sync(0xffffffffu, v1, 16);
            if (lane == 0) {
                sh_y[j][warp * 2 + 0] = (v0 / (1.0f + __expf(-v0))) * u0;
                sh_y[j][warp * 2 + 1] = (v1 / (1.0f + __expf(-v1))) * u1;
            }
        }
    }
    __syncthreads();

    if (warp < cnt && lane < 16) {
        const float yv = sh_y[warp][lane];
        float amax = fabsf(yv);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffu, amax, 8));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffu, amax, 1));
        const float d = amax / 127.0f;
        const float inv_d = d > 0.f ? (127.0f / amax) : 0.f;
        const int ej = grp_e[g * GRP_MAX + warp];
        if (lane == 0) ds16[(size_t) ej * (NFF / 16) + blk16] = d;
        const int qv = max(-127, min(127, __float2int_rn(yv * inv_d)));
        dq16[(size_t) ej * NFF + blk16 * 16 + lane] = (signed char) qv;
    }
}

// 2-rows-per-subwarp (64 rows per 256-thread block!) `exp_down_2r_grouped`:
// Launch grid: `dim3(NE / 64, max_groups)` = `dim3(40, max_groups)`, Block: 256
__global__ void __launch_bounds__(256) exp_down_2r_grouped(
    const unsigned char* __restrict__ b0,
    const int* __restrict__ grp_slot, const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_e, const int* __restrict__ n_groups,
    const signed char* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ out) {

    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qrow = lane >> 3;  // 0..3
    const int qlane = lane & 7;  // 0..7
    const int row0 = blockIdx.x * 64 + warp * 8 + qrow * 2;
    const int cnt = grp_cnt[g];

    int e_idx[GRP_MAX];
    float acc0[GRP_MAX], acc1[GRP_MAX];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        e_idx[j] = (j < cnt) ? grp_e[g * GRP_MAX + j] : 0;
        acc0[j] = 0.f;
        acc1[j] = 0.f;
    }

    const unsigned char* blob = b0 + (size_t) grp_slot[g] * PER;
    const unsigned char* base = blob + 2 * (size_t) NFF * GU_ROW;
    const unsigned char* r_ptr0 = base + (size_t) row0 * D_ROW;
    const unsigned char* r_ptr1 = r_ptr0 + D_ROW;

    if (cnt == 1) {
        const signed char* xq0 = xq + (size_t) e_idx[0] * NFF;
        const float* xs0 = xs + (size_t) e_idx[0] * (NFF / 16);
        #pragma unroll
        for (int it = 0; it < 5; ++it) {
            const int sub = it * 8 + qlane;
            const int b = sub >> 2;
            const int s = sub & 3;
            const int4 av = *(const int4*) (xq0 + sub * 16);
            const float xsc = xs0[sub];

            const unsigned char* db0 = r_ptr0 + b * 36;
            const float ds0 = ue4m3(__ldg(db0 + s));
            const uint32_t* dp0 = (const uint32_t*) (db0 + 4 + s * 8);
            int dw0a, dw1a, dw2a, dw3a;
            unpack_nvfp4_sub16_reg(__ldg(dp0), __ldg(dp0 + 1), dw0a, dw1a, dw2a, dw3a);

            const unsigned char* db1 = r_ptr1 + b * 36;
            const float ds1 = ue4m3(__ldg(db1 + s));
            const uint32_t* dp1 = (const uint32_t*) (db1 + 4 + s * 8);
            int dw0b, dw1b, dw2b, dw3b;
            unpack_nvfp4_sub16_reg(__ldg(dp1), __ldg(dp1 + 1), dw0b, dw1b, dw2b, dw3b);

            const int d0 = __dp4a(dw0a, av.x, __dp4a(dw1a, av.y, __dp4a(dw2a, av.z, __dp4a(dw3a, av.w, 0))));
            const int d1 = __dp4a(dw0b, av.x, __dp4a(dw1b, av.y, __dp4a(dw2b, av.z, __dp4a(dw3b, av.w, 0))));
            acc0[0] = fmaf(ds0 * xsc, (float) d0, acc0[0]);
            acc1[0] = fmaf(ds1 * xsc, (float) d1, acc1[0]);
        }
    } else {
        #pragma unroll
        for (int it = 0; it < 5; ++it) {
            const int sub = it * 8 + qlane;
            const int b = sub >> 2;
            const int s = sub & 3;

            const unsigned char* db0 = r_ptr0 + b * 36;
            const float ds0 = ue4m3(__ldg(db0 + s));
            const uint32_t* dp0 = (const uint32_t*) (db0 + 4 + s * 8);
            int dw0a, dw1a, dw2a, dw3a;
            unpack_nvfp4_sub16_reg(__ldg(dp0), __ldg(dp0 + 1), dw0a, dw1a, dw2a, dw3a);

            const unsigned char* db1 = r_ptr1 + b * 36;
            const float ds1 = ue4m3(__ldg(db1 + s));
            const uint32_t* dp1 = (const uint32_t*) (db1 + 4 + s * 8);
            int dw0b, dw1b, dw2b, dw3b;
            unpack_nvfp4_sub16_reg(__ldg(dp1), __ldg(dp1 + 1), dw0b, dw1b, dw2b, dw3b);

            #pragma unroll
            for (int j = 0; j < GRP_MAX; ++j) {
                if (j < cnt) {
                    const int ej = e_idx[j];
                    const int4 av = *(const int4*) (xq + (size_t) ej * NFF + sub * 16);
                    const float xsc = xs[(size_t) ej * (NFF / 16) + sub];
                    const int d0 = __dp4a(dw0a, av.x, __dp4a(dw1a, av.y, __dp4a(dw2a, av.z, __dp4a(dw3a, av.w, 0))));
                    const int d1 = __dp4a(dw0b, av.x, __dp4a(dw1b, av.y, __dp4a(dw2b, av.z, __dp4a(dw3b, av.w, 0))));
                    acc0[j] = fmaf(ds0 * xsc, (float) d0, acc0[j]);
                    acc1[j] = fmaf(ds1 * xsc, (float) d1, acc1[j]);
                }
            }
        }
    }

    const float sc_d = ((const float*) (blob + PER - 16))[2];
    #pragma unroll
    for (int j = 0; j < GRP_MAX; ++j) {
        if (j < cnt) {
            float v0 = acc0[j], v1 = acc1[j];
            v0 += __shfl_xor_sync(0xffffffffu, v0, 4);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 4);
            v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
            v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
            if (qlane == 0) {
                out[(size_t) e_idx[j] * NE + row0] = v0 * sc_d;
                out[(size_t) e_idx[j] * NE + row0 + 1] = v1 * sc_d;
            }
        }
    }
}

// Helpers for Unlocked Tensor-Core `cublasGemmEx` Batched Prefill (`T >= 16`)
__global__ void dequant_q8p_to_fp16(const uint8_t* __restrict__ codes, const unsigned short* __restrict__ scales,
                                    __half* __restrict__ out, int rows, int cols) {
    const size_t total = (size_t) rows * cols;
    const size_t idx4 = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 >= total) return;
    const size_t r = idx4 / cols, c = idx4 - r * cols;
    const __half sc = *(__half*) &scales[r * (cols / 32) + (c >> 5)];
    const char4 q4 = *(const char4*) (codes + idx4);
    __half2* o2 = (__half2*) (out + idx4);
    const __half2 sc2(sc, sc);
    o2[0] = __hmul2(__floats2half2_rn((float) q4.x, (float) q4.y), sc2);
    o2[1] = __hmul2(__floats2half2_rn((float) q4.z, (float) q4.w), sc2);
}

__global__ void bf16_to_fp16_vec(const unsigned short* __restrict__ src, __half* __restrict__ dst, size_t n) {
    const size_t i = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;
    const float f0 = db2f(src[i]);
    const float f1 = (i + 1 < n) ? db2f(src[i + 1]) : 0.f;
    *(__half2*) (dst + i) = __floats2half2_rn(f0, f1);
}

__global__ void f32_to_fp16_vec(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    const size_t i = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;
    const float f0 = src[i];
    const float f1 = (i + 1 < n) ? src[i + 1] : 0.f;
    *(__half2*) (dst + i) = __floats2half2_rn(f0, f1);
}

__global__ void restore_bf16_to_f32_kernel(const unsigned short* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = db2f(src[i]);
}

// Multi-row GPU greedy argmax over `nvocab` (248,320) logits -> copies 28 bytes instead of 7.0 MB over PCIe!
__global__ void row_argmax_multi_kernel(const float* __restrict__ logits, int nvocab, int* __restrict__ out_ids) {
    const int row = blockIdx.x;
    const float* l = logits + (size_t) row * nvocab;
    const int tid = threadIdx.x; // 1024 threads (32 warps)
    const int lane = tid & 31, warp = tid >> 5;
    float bv = -1e30f;
    int bi = 0;
    for (int i = tid; i < nvocab; i += 1024) {
        const float v = l[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        bv = sv[lane];
        bi = si[lane];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out_ids[row] = bi;
    }
}

// Single-pass GDDR6 Top-16 + Greedy Argmax over `nvocab` (248,320) logits (< 0.025 ms, zero L2 thrashing!)
// Block: 512 threads (each thread tracks its top-4 in 8 scalar regs -> 2048 candidates in 16 KB shared memory!)
__global__ void row_top16_multi_kernel(const float* __restrict__ logits, int nvocab,
                                       int* __restrict__ out_greedy,
                                       float* __restrict__ out_top16_v,
                                       int* __restrict__ out_top16_i) {
    const int row = blockIdx.x;
    const float* l = logits + (size_t) row * nvocab;
    const int tid = threadIdx.x; // 512 threads (16 warps)
    const int lane = tid & 31, warp = tid >> 5;

    float v0 = -1e30f, v1 = -1e30f, v2 = -1e30f, v3 = -1e30f;
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    for (int i = tid; i < nvocab; i += 512) {
        const float v = l[i];
        if (v > v3) {
            if (v > v0)      { v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = v; i0 = i; }
            else if (v > v1) { v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v;  i1 = i; }
            else if (v > v2) { v3 = v2; i3 = i2; v2 = v;  i2 = i; }
            else             { v3 = v;  i3 = i; }
        }
    }

    __shared__ float cand_v[2048];
    __shared__ int cand_i[2048];
    __shared__ float sv[16], s_out_v[16];
    __shared__ int si[16], s_out_i[16];
    cand_v[tid * 4 + 0] = v0; cand_i[tid * 4 + 0] = i0;
    cand_v[tid * 4 + 1] = v1; cand_i[tid * 4 + 1] = i1;
    cand_v[tid * 4 + 2] = v2; cand_i[tid * 4 + 2] = i2;
    cand_v[tid * 4 + 3] = v3; cand_i[tid * 4 + 3] = i3;
    __syncthreads();

    for (int p = 0; p < 16; ++p) {
        float bv = -1e30f;
        int bi = nvocab, bslot = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int s = u * 512 + tid;
            const float cv = cand_v[s];
            const int ci = cand_i[s];
            if (cv > bv || (cv == bv && ci < bi)) { bv = cv; bi = ci; bslot = s; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int os = __shfl_xor_sync(0xffffffffu, bslot, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bslot = os; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = (bi << 11) | bslot; }
        __syncthreads();
        if (warp == 0) {
            bv = (lane < 16) ? sv[lane] : -1e30f;
            const int packed = (lane < 16) ? si[lane] : ((nvocab << 11) | 0);
            bi = packed >> 11;
            bslot = packed & 2047;
            #pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                const int os = __shfl_xor_sync(0xffffffffu, bslot, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bslot = os; }
            }
            if (lane == 0) {
                s_out_v[p] = bv;
                s_out_i[p] = bi;
                cand_v[bslot] = -1e30f;
            }
        }
        __syncthreads();
    }
    if (tid == 0 && out_greedy) out_greedy[row] = s_out_i[0];
    if (tid < 16) {
        if (out_top16_v) out_top16_v[row * 16 + tid] = s_out_v[tid];
        if (out_top16_i) out_top16_i[row * 16 + tid] = s_out_i[tid];
    }
}
