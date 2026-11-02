// wzy_kernels.cuh -- SM75 (2x RTX 2080 Ti 22GB) PP2 Dual-GPU Kernels for Qwen3.8-Flash-Next-NVFP4
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

static const int NE = 2560, HC = 4, HCLR = 320, HCD = HC * NE;
static const int QH = 24, KVH = 2, HD = 256, QD = QH * HD, KVD = KVH * HD;
static const int GH = 16, VH = 48, ST = 128, GQ = GH * ST, GV = VH * ST, QKV = 2 * GQ + GV;
static const int NFF = 640, NEXP = 512, TOPK = 10, NVOCAB = 248320;
static const int NROT = 64;
static const float ROPE_THETA = 1e7f, EPS = 1e-6f;
static const size_t PER = 2764816;              // one NVFP4 expert blob (2.76 MB)
static const int GU_ROW = NE / 64 * 36, D_ROW = NFF / 64 * 36;
static const int PLE_HEAD = 160, PLE_HEADS = 16, PLE_NG = 3, PLE_HPN = 8, PLE_KERN = 4;
static const int PLE_HIST = (PLE_KERN - 1) * PLE_NG;      // 9
static const int PLE_EOS = 248044;

__device__ __forceinline__ float dh2f(unsigned short h) { return __half2float(*(__half*) &h); }
__device__ __forceinline__ float db2f(unsigned short h) { return __uint_as_float((unsigned int) h << 16); }
__device__ __constant__ float kUe4m3[128];
__device__ __constant__ unsigned short kPair[256];
__device__ __forceinline__ float ue4m3(unsigned x) { return kUe4m3[x & 0x7F]; }

__device__ __forceinline__ float dwsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ int pack4(const unsigned char* b, int high) {
    const int p0 = kPair[b[0]], p1 = kPair[b[1]], p2 = kPair[b[2]], p3 = kPair[b[3]];
    const int lo = (p0 & 0xFF) | ((p1 & 0xFF) << 8) | ((p2 & 0xFF) << 16) | ((p3 & 0xFF) << 24);
    const int hi = (p0 >> 8) | ((p1 >> 8) << 8) | ((p2 >> 8) << 16) | ((p3 >> 8) << 24);
    return high ? hi : lo;
}
__device__ __forceinline__ int pack4t(const unsigned short* tab, const unsigned char* b, int high) {
    const int p0 = tab[b[0]], p1 = tab[b[1]], p2 = tab[b[2]], p3 = tab[b[3]];
    const int lo = (p0 & 0xFF) | ((p1 & 0xFF) << 8) | ((p2 & 0xFF) << 16) | ((p3 & 0xFF) << 24);
    const int hi = (p0 >> 8) | ((p1 >> 8) << 8) | ((p2 >> 8) << 16) | ((p3 >> 8) << 24);
    return high ? hi : lo;
}

__global__ void quant_act(const float* __restrict__ x, signed char* __restrict__ q32, float* __restrict__ s32,
                          signed char* __restrict__ q16, float* __restrict__ s16, int n) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g * 16 >= n) return;
    const float* xb = x + g * 16;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float d = amax / 127.0f;
    s16[g] = d;
    signed char tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int v = d > 0.f ? __float2int_rn(xb[i] / d) : 0;
        tmp[i] = (signed char) (v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) q16[g * 16 + i] = tmp[i];
    if ((g & 1) == 0) {
        float am2 = amax;
        #pragma unroll
        for (int i = 0; i < 16; ++i) am2 = fmaxf(am2, fabsf(xb[16 + i]));
        const float d2 = am2 / 127.0f;
        s32[g >> 1] = d2;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int v = d2 > 0.f ? __float2int_rn(xb[16 + i] / d2) : 0;
            q32[(g >> 1) * 32 + 16 + i] = (signed char) (v > 127 ? 127 : (v < -127 ? -127 : v));
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int v = d2 > 0.f ? __float2int_rn(xb[i] / d2) : 0;
            q32[(g >> 1) * 32 + i] = (signed char) (v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

__global__ void gemv_q8p(const uint8_t* __restrict__ codes, const unsigned short* __restrict__ scales,
                         const signed char* __restrict__ xq, const float* __restrict__ xs,
                         float* __restrict__ y, int rows, int cols) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const int nb = cols / 32;
    const uint8_t* c = codes + (size_t) warp * cols;
    const unsigned short* sc = scales + (size_t) warp * nb;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int4 a = *(const int4*) (c + (size_t) b * 32);
        const int4 bq = *(const int4*) (c + (size_t) b * 32 + 16);
        const int* qx = (const int*) (xq + b * 32);
        int isum = 0;
        isum = __dp4a(a.x, qx[0], isum);
        isum = __dp4a(a.y, qx[1], isum);
        isum = __dp4a(a.z, qx[2], isum);
        isum = __dp4a(a.w, qx[3], isum);
        isum = __dp4a(bq.x, qx[4], isum);
        isum = __dp4a(bq.y, qx[5], isum);
        isum = __dp4a(bq.z, qx[6], isum);
        isum = __dp4a(bq.w, qx[7], isum);
        acc += dh2f(sc[b]) * xs[b] * (float) isum;
    }
    acc = dwsum(acc);
    if (lane == 0) y[warp] = acc;
}

__global__ void gemv_bf16(const unsigned short* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int rows, int cols) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const unsigned short* w = W + (size_t) warp * cols;
    float acc = 0.f;
    for (int i = lane; i < cols; i += 32) acc += db2f(w[i]) * x[i];
    acc = dwsum(acc);
    if (lane == 0) y[warp] = acc;
}

__global__ void gemv_bf16_split(const unsigned short* __restrict__ W, const float* __restrict__ x,
                                float* __restrict__ y, int rows, int cols, int SP) {
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row = gwarp / SP, part = gwarp % SP;
    if (row >= rows) return;
    const unsigned short* w = W + (size_t) row * cols;
    const int chunk = (cols + SP - 1) / SP;
    const int lo = part * chunk, hi = min(cols, lo + chunk);
    float acc = 0.f;
    for (int i = lo + lane; i < hi; i += 32) acc += db2f(w[i]) * x[i];
    acc = dwsum(acc);
    if (lane == 0) atomicAdd(&y[row], acc);
}

__global__ void gemv_f32(const float* __restrict__ W, const float* __restrict__ x,
                         float* __restrict__ y, int rows, int cols) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const float* w = W + (size_t) warp * cols;
    float acc = 0.f;
    for (int i = lane; i < cols; i += 32) acc += w[i] * x[i];
    acc = dwsum(acc);
    if (lane == 0) y[warp] = acc;
}

__global__ void group_norm(const float* __restrict__ R, const float* __restrict__ g,
                           float* __restrict__ out, int n_embd, float eps) {
    const int c = blockIdx.x;
    const float* r = R + (size_t) c * n_embd;
    const float* gc = g + (size_t) c * n_embd;
    float* o = out + (size_t) c * n_embd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) ss += r[i] * r[i];
    __shared__ float sh[32];
    ss = dwsum(ss);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += sh[i];
        sh[0] = rsqrtf(t / (float) n_embd + eps);
    }
    __syncthreads();
    const float sc = sh[0];
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) o[i] = r[i] * sc * gc[i];
}

__global__ void gr_gate(const float* __restrict__ xn, const float* __restrict__ gg,
                        float* __restrict__ gated, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gated[i] = xn[i] / (1.f + __expf(-gg[i]));
}
__global__ void gr_mean(const float* __restrict__ gated, float* __restrict__ mixed, int n_embd, int hc,
                        float div) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float s = 0.f;
    for (int c = 0; c < hc; ++c) s += gated[(size_t) c * n_embd + d];
    mixed[d] = s / div;
}
__global__ void gr_write_k(const float* __restrict__ R, const float* __restrict__ blk,
                           const float* __restrict__ inj, float* __restrict__ out, int n_embd, int hc,
                           float wscale, float hdiv) {
    extern __shared__ float sw[];
    if ((int) threadIdx.x < hc) sw[threadIdx.x] = wscale / (1.f + __expf(-inj[threadIdx.x] / hdiv));
    __syncthreads();
    const int n = hc * n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int c = i / n_embd, d = i - c * n_embd;
        out[i] = R[i] + blk[d] * sw[c];
    }
}

__global__ void silu_mul(const float* __restrict__ a, const float* __restrict__ b,
                         float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] / (1.f + __expf(-a[i]))) * b[i];
}
__global__ void silu_inplace(float* __restrict__ x, int n, float scale = 1.f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = x[i] * scale; x[i] = v / (1.f + __expf(-v)); }
}

// GDN: conv (silu) + L2 + delta rule (with exact 1/sqrt(S) readout scale) + out_norm
__global__ void gdn_conv(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                         float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    float acc = 0.f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * kW[c * dc + i];
    acc += x[c] * kW[c * dc + dc - 1];
    out[c] = acc / (1.f + __expf(-acc));
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

__global__ void l2_norm(float* __restrict__ x, int cols, float eps) {
    const int r = blockIdx.x;
    float* p = x + (size_t) r * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += p[i] * p[i];
    ss = dwsum(ss);
    __shared__ float sh[32];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += sh[i];
        sh[0] = rsqrtf(t + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) p[i] *= sh[0];
}

__global__ void gdn_gate(const float* __restrict__ alpha, const float* __restrict__ dt,
                         const float* __restrict__ a, const float* __restrict__ beta,
                         float* __restrict__ decay, float* __restrict__ bt, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = alpha[i] + dt[i];
    decay[i] = __expf((v > 20.f ? v : log1pf(__expf(v))) * a[i]);
    bt[i] = 1.f / (1.f + __expf(-beta[i]));
}

__global__ void gdn_step(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                         const float* __restrict__ v, const float* __restrict__ decay,
                         const float* __restrict__ bt, float* __restrict__ o, int S, int hk, int hv) {
    extern __shared__ float sh[];
    float* ks = sh; float* qs = sh + S; float* sc = sh + 2 * S;
    const int h = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int src = h % hk;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { ks[i] = k[src * S + i]; qs[i] = q[src * S + i]; }
    if (threadIdx.x == 0) { sc[0] = decay[h]; sc[1] = bt[h]; }
    __syncthreads();
    if (j >= S) return;
    float* col = state + (size_t) h * S + j;
    const size_t stride = (size_t) hv * S;
    float sk = 0.f;
    for (int i = 0; i < S; ++i) {
        const float s = col[(size_t) i * stride] * sc[0];
        col[(size_t) i * stride] = s;
        sk += s * ks[i];
    }
    const float d = (v[(size_t) h * S + j] - sk) * sc[1];
    float dot = 0.f;
    for (int i = 0; i < S; ++i) {
        const float s = col[(size_t) i * stride] + ks[i] * d;
        col[(size_t) i * stride] = s;
        dot += s * qs[i];
    }
    // Critical fix: 1/sqrt(S) readout scale before RMSNorm (matches fused_gdn.cu L63)
    o[(size_t) h * S + j] = dot * rsqrtf((float) S);
}

__global__ void gdn_out_norm(const float* __restrict__ o, const float* __restrict__ z,
                             const float* __restrict__ g, float* __restrict__ out, int S, float eps) {
    const int h = blockIdx.x;
    const float* oh = o + (size_t) h * S;
    float* ob = out + (size_t) h * S;
    float ss = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) ss += oh[i] * oh[i];
    ss = dwsum(ss);
    __shared__ float sh[32];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += sh[i];
        sh[0] = rsqrtf(t / (float) S + eps);
    }
    __syncthreads();
    const float sc = sh[0];
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        const float zz = z[(size_t) h * S + i];
        ob[i] = oh[i] * sc * g[i] * (1.f / (1.f + __expf(-zz)));
    }
}

// QSA Attention
__global__ void head_norm(float* __restrict__ x, const float* __restrict__ g, int hd, int stride,
                          float eps) {
    const int h = blockIdx.x;
    float* p = x + (size_t) h * stride;
    __shared__ float sc;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int i = 0; i < hd; ++i) ss += p[i] * p[i];
        sc = rsqrtf(ss / (float) hd + eps);
    }
    __syncthreads();
    const float s = sc;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) p[i] *= s * g[i];
}

__global__ void rope_apply(float* __restrict__ v, int heads, const int* pos_dev, int hd, int nrot, float theta) {
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= nrot / 2) return;
    float* b = v + (size_t) h * hd;
    const double inv = pow((double) theta, -2.0 * (double) i / (double) nrot);
    const double ang = (double) (*pos_dev) * inv;
    const float cs = (float) cos(ang), sn = (float) sin(ang);
    const float a = b[i], d = b[i + nrot / 2];
    b[i] = a * cs - d * sn;
    b[i + nrot / 2] = a * sn + d * cs;
}

__global__ void kv_append(float* __restrict__ kc, float* __restrict__ vc,
                          const float* __restrict__ k, const float* __restrict__ v,
                          const int* ctx_dev, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t at = (size_t) (*ctx_dev) * n + i;
    kc[at] = k[i];
    vc[at] = v[i];
}

// Parallel-across-positions QSA decode attention using device score buffer (no 48KB shared mem limit, bitwise identical per-p dot)
__global__ void qsa_attend_fast(const float* __restrict__ qg, const float* __restrict__ kc,
                                const float* __restrict__ vc, float* __restrict__ sc_buf, int ctx_max,
                                float* __restrict__ out, const int* ctx_dev, int qh, int kvh, int hd,
                                int qstride, int goff) {
    const int ctx = *ctx_dev;
    const int npos = min(ctx + 1, ctx_max);
    const int h = blockIdx.x;
    const int kv = h / (qh / kvh);
    const float scale = 1.f / sqrtf((float) hd);
    float* sc = sc_buf + (size_t) h * ctx_max;

    // 256 threads compute independent positions p in parallel; each position's 256-dim dot is sequential
    for (int p = threadIdx.x; p < npos; p += blockDim.x) {
        const float* kp = kc + ((size_t) p * kvh + kv) * hd;
        float a = 0.f;
        for (int d = 0; d < hd; ++d) a += qg[(size_t) h * qstride + d] * kp[d];
        sc[p] = a * scale;
    }
    __syncthreads();

    __shared__ float ssum;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int p = 0; p < npos; ++p) m = fmaxf(m, sc[p]);
        float t = 0.f;
        for (int p = 0; p < npos; ++p) { sc[p] = expf(sc[p] - m); t += sc[p]; }
        ssum = t;
    }
    __syncthreads();

    const float inv_s = 1.f / ssum;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < npos; ++p)
            acc += (sc[p] * inv_s) * vc[((size_t) p * kvh + kv) * hd + d];
        const float g = qg[(size_t) h * qstride + goff + d];
        out[(size_t) h * hd + d] = acc / (1.f + expf(-g));
    }
}

// Router top-k
__global__ void router_topk(const float* __restrict__ logits, int n, int k, int* __restrict__ ids,
                            float* __restrict__ wts) {
    extern __shared__ double ex[];
    float* p = (float*) (ex + n);
    char* taken = (char*) (p + n);
    float mx = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, logits[i]);
    __shared__ float sh[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = -1e30f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t = fmaxf(t, sh[i]);
        sh[0] = t;
    }
    __syncthreads();
    mx = sh[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) ex[i] = (double) __expf(logits[i] - mx);
    __syncthreads();
    __shared__ double ssum;
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += ex[i];
        ssum = s;
        for (int i = 0; i < n; ++i) taken[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = (float) (ex[i] / ssum);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < k; ++j) {
            float bv = -1e30f;
            int bi = n;
            for (int i = 0; i < n; ++i) {
                if (taken[i]) continue;
                if (p[i] > bv) { bv = p[i]; bi = i; }
            }
            ids[j] = bi;
            wts[j] = bv;
            taken[bi] = 1;
        }
        double s2 = 0.0;
        for (int i = 0; i < k; ++i) s2 += (double) wts[i];
        const double sc2 = s2 > 6.103515625e-05 ? s2 : 6.103515625e-05;
        for (int i = 0; i < k; ++i) wts[i] = (float) ((double) wts[i] / sc2);
    }
}

// Direct O(1) LUT slot lookup:
// lut[global_key] >= 0  -> VRAM slot in xblobs (local GPU VRAM at 550 GB/s)
// lut[global_key] <  0  -> Pinned host pack at d_pack + global_key * PER (UVA zero-copy over PCIe Gen3 x16)
__global__ void map_slots_lut(const int* __restrict__ ids, int k, const int* __restrict__ lut,
                              int layer, int* __restrict__ slots, int* __restrict__ misses,
                              int* __restrict__ miss_key) {
    const int i = threadIdx.x;
    if (i >= k) return;
    const int gkey = layer * NEXP + ids[i];
    const int s = lut[gkey];
    slots[i] = (s >= 0) ? s : -(1 + gkey);
    if (s < 0) {
        const int m = atomicAdd(misses, 1);
        if (m < 512) miss_key[m] = gkey;
    }
}

#define SPLIT 4
__device__ __forceinline__ const unsigned char* blob_at_hybrid(
    const unsigned char* vram_blobs, const unsigned char* host_pack, int slot) {
    return (slot >= 0) ? (vram_blobs + (size_t) slot * PER)
                       : (host_pack + (size_t) (-1 - slot) * PER);
}

__global__ void exp_scales(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                           const int* __restrict__ ids, int k, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= k) return;
    const float* tail = (const float*) (blob_at_hybrid(b0, hpack, ids[e]) + PER - 16);
    out[3 * e + 0] = tail[0];
    out[3 * e + 1] = tail[1];
    out[3 * e + 2] = tail[2];
}

__global__ void exp_gu(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                       const int* __restrict__ ids, int nb_experts,
                       const signed char* __restrict__ xq, const float* __restrict__ xs,
                       float* __restrict__ ff, const float* __restrict__ scales, int nrows, int nvals,
                       int row_bytes) {
    __shared__ unsigned short tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = kPair[i];
    __syncthreads();
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int e = gwarp / nrows;
    if (e >= nb_experts) return;
    const int row = gwarp - e * nrows;
    const unsigned char* base = blob_at_hybrid(b0, hpack, ids[e]);
    const unsigned char* g = base + (size_t) row * row_bytes;
    const unsigned char* u = base + (size_t) (nrows + row) * row_bytes;
    const int nb = nvals / 64;
    float ag = 0.f, au = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int* a = (const int*) (xq + b * 64);
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const unsigned char* blk = (pass ? u : g) + b * 36;
            const unsigned int ds = *(const unsigned int*) blk;
            const unsigned char* q = blk + 4;
            float acc = 0.f;
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int i0 = __dp4a(pack4t(tab, q + s * 8 + 0, 0), a[s * 4 + 0], 0);
                const int i1 = __dp4a(pack4t(tab, q + s * 8 + 4, 0), a[s * 4 + 1], 0);
                const int i2 = __dp4a(pack4t(tab, q + s * 8 + 0, 1), a[s * 4 + 2], 0);
                const int i3 = __dp4a(pack4t(tab, q + s * 8 + 4, 1), a[s * 4 + 3], 0);
                acc += ue4m3((ds >> (8 * s)) & 0xFF) * xs[b * 4 + s] * (float) (i0 + i1 + i2 + i3);
            }
            if (pass == 0) ag += acc; else au += acc;
        }
    }
    ag = dwsum(ag) * scales[3 * e + 0];
    au = dwsum(au) * scales[3 * e + 1];
    if (lane == 0) {
        ff[(size_t) e * nrows * 3 + (size_t) nrows + row] = ag;
        ff[(size_t) e * nrows * 3 + (size_t) 2 * nrows + row] = au;
    }
}

__global__ void exp_gu_epi(float* __restrict__ ff, float* __restrict__ act, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = i / NFF, r = i - e * NFF;
    float* b = ff + (size_t) e * NFF * 3;
    const float g = b[NFF + r], u = b[2 * NFF + r];
    const float a = (g / (1.f + __expf(-g))) * u;
    b[r] = a;
    act[i] = a;
}

__global__ void exp_down(const unsigned char* __restrict__ b0, const unsigned char* __restrict__ hpack,
                         const int* __restrict__ ids, int nb_experts,
                         const signed char* __restrict__ xq, const float* __restrict__ xs,
                         float* __restrict__ out, const float* __restrict__ scales, int nrows, int nvals,
                         int row_bytes) {
    __shared__ unsigned short tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = kPair[i];
    __syncthreads();
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int e = gwarp / nrows;
    if (e >= nb_experts) return;
    const int row = gwarp - e * nrows;
    const unsigned char* base = blob_at_hybrid(b0, hpack, ids[e]) + 2 * (size_t) NFF * GU_ROW;
    const unsigned char* r = base + (size_t) row * row_bytes;
    const signed char* xqe = xq + (size_t) e * nvals;
    const float* xse = xs + (size_t) e * (nvals / 16);
    const int nb = nvals / 64;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char* blk = r + b * 36;
        const unsigned int ds = *(const unsigned int*) blk;
        const unsigned char* q = blk + 4;
        const int* a = (const int*) (xqe + b * 64);
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int i0 = __dp4a(pack4t(tab, q + s * 8 + 0, 0), a[s * 4 + 0], 0);
            const int i1 = __dp4a(pack4t(tab, q + s * 8 + 4, 0), a[s * 4 + 1], 0);
            const int i2 = __dp4a(pack4t(tab, q + s * 8 + 0, 1), a[s * 4 + 2], 0);
            const int i3 = __dp4a(pack4t(tab, q + s * 8 + 4, 1), a[s * 4 + 3], 0);
            acc += ue4m3((ds >> (8 * s)) & 0xFF) * xse[b * 4 + s] * (float) (i0 + i1 + i2 + i3);
        }
    }
    acc = dwsum(acc) * scales[3 * e + 2];
    if (lane == 0) out[(size_t) e * nrows + row] = acc;
}

__global__ void moe_combine(const float* __restrict__ parts, const float* __restrict__ wts,
                            const float* __restrict__ shared, float* __restrict__ out, int n, int k) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    float s = shared[d];
    for (int e = 0; e < k; ++e) s += wts[e] * parts[(size_t) e * n + d];
    out[d] = s;
}
__global__ void sigmoid_scale_dev(float* __restrict__ y, const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = 1.f / (1.f + __expf(-*g));
    y[i] *= s;
}
__global__ void dot1(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += a[i] * b[i];
    s = dwsum(s);
    __shared__ float sh[32];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += sh[i];
        out[0] = t;
    }
}

// PLE (layer 1): gamma is [c * n_embd + d] (fixed from transposed [d * hc + c])
__global__ void ple_group(const float* __restrict__ x, const float* __restrict__ gamma,
                          float* __restrict__ y, int n_embd, int hc, float eps) {
    const int c = blockIdx.x;
    const float* xc = x + (size_t) c * n_embd;
    float* yc = y + (size_t) c * n_embd;
    __shared__ float red[32];
    float ss = 0.f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) ss += xc[d] * xc[d];
    ss = dwsum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
        red[0] = rsqrtf(t / (float) n_embd + eps);
    }
    __syncthreads();
    const float rs = red[0];
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x)
        yc[d] = xc[d] * rs * gamma[(size_t) c * n_embd + d];
}

__global__ void ple_gate(const float* __restrict__ key, const float* __restrict__ query,
                         float* __restrict__ gate, int hc, int n_embd) {
    const int c = blockIdx.x;
    const float* kc = key + (size_t) c * n_embd;
    const float* qc = query + (size_t) c * n_embd;
    __shared__ float red[32];
    float acc = 0.f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) acc += kc[d] * qc[d];
    acc = dwsum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
        t /= sqrtf((float) n_embd);
        const float sg = t < 0.f ? -1.f : 1.f;
        gate[c] = 1.f / (1.f + __expf(-sg * sqrtf(fmaxf(fabsf(t), 1e-6f))));
    }
}

__global__ void ple_gv(const float* __restrict__ value, const float* __restrict__ gate,
                       float* __restrict__ gv, int hcd, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hcd) return;
    gv[i] = value[i % n_embd] * gate[i / n_embd];
}

__global__ void ple_conv(const float* __restrict__ norm, const float* __restrict__ cw,
                         float* __restrict__ hist, const float* __restrict__ gv,
                         float* __restrict__ R, int hcd, int ng, int kern) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hcd) return;
    const int H = (kern - 1) * ng;
    float acc = 0.f;
    for (int k = 0; k < kern; ++k) {
        const float src = (k == kern - 1) ? norm[c]
                                         : hist[(size_t) (ng * k) + (size_t) H * c];
        acc += cw[k + kern * c] * src;
    }
    for (int i = 0; i + 1 < H; ++i)
        hist[(size_t) i + (size_t) H * c] = hist[(size_t) (i + 1) + (size_t) H * c];
    hist[(size_t) (H - 1) + (size_t) H * c] = norm[c];
    R[c] += gv[c] + acc / (1.f + __expf(-acc));
}
