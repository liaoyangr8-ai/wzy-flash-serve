#!/usr/bin/env python3
"""F1 (README 7.16): fuse the router phase into ONE kernel per token.

Replaces, per layer:
    dense_bf16(w.g_inp, mixed2, slog, NEXP, NE, T)          // 64-block gemm, warp-per-row
    router_top10_fast_multi<<<T,256>>>(slog, sids, swts)     // softmax + top-10
    launch_gemm_bf16(w.s_gate_inp, mixed2, sgate, 1, NE, T)  // rows=1 gemm (1 warp active!)
with:
    router_gate_top10_fused<<<T, 512>>>(...)                 // 16 warps x 32 rows each

Bit-equality contract (the equivalence criterion is byte-identical token streams):
  * dot products: per row e, lane-strided j loop over cols/8 uint4 pairs, the same 8-product
    expression and the same dwsum warp reduction as gemm_bf16_t -- identical per-(row,token) order;
  * sgate: warp 0 of each block reproduces the rows=1 gemm order for its token;
  * softmax/top-10: copied verbatim from router_top10_fast_multi, reading the shared logits instead
    of global slog (same values).
slog/sgate are still written (same layouts) so anything downstream is untouched.
"""
import shutil

K = "/home/liaoyang/wzy-flash-serve/src/wzy_kernels_v2.cuh"
S = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"

k = open(K, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(K, "/tmp/wzy_kernels_v2.cuh.bak-f1")
k = k.replace("\r\n", "\n")

s = open(S, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(S, "/tmp/wzy_stage_v2.cuh.bak-f1")
s = s.replace("\r\n", "\n")


def sub_once(txt, old, new, what):
    n = txt.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:200])
    return txt.replace(old, new)


FUSED = r"""
// ==================================================================
// F1 (README 7.16): router gate gemm + shared-expert gate gemm + top-10 router in ONE launch.
// Grid = T blocks x 512 threads (16 warps); warp w computes expert rows w, w+16, ... with the SAME
// lane-strided accumulation order as gemm_bf16_t (bit-identical logits), warp 0 also reproduces the
// rows=1 sgate dot product, then the first 8 warps run the router_top10 softmax/argmax verbatim
// over the shared logits.
// ==================================================================
__global__ void router_gate_top10_fused(const unsigned short* __restrict__ Wg,
                                        const unsigned short* __restrict__ Ws,
                                        const float* __restrict__ x,
                                        float* __restrict__ slog,
                                        float* __restrict__ sgate,
                                        int* __restrict__ ids, float* __restrict__ wts,
                                        int cols) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;            // 512 threads
    const int lane = tid & 31, warp = tid >> 5;
    const int NW = blockDim.x >> 5;         // 16
    const float* xt = x + (size_t) t * cols;
    const int n8 = cols / 8;

    __shared__ float sp[NEXP];
    __shared__ float wr[8];
    __shared__ float sg1;

    for (int e = warp; e < NEXP; e += NW) {
        const uint4* w4 = (const uint4*) (Wg + (size_t) e * cols);
        float acc = 0.f;
        for (int j = lane; j < n8; j += 32) {
            const uint4 wv = __ldg(w4 + j);
            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
            const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
            const float4 xa = *(const float4*) (xt + j * 8);
            const float4 xb = *(const float4*) (xt + j * 8 + 4);
            acc += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                 + w4f * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
        acc = dwsum(acc);
        if (lane == 0) sp[e] = acc;
    }
    if (warp == 0) {   // rows=1 sgate dot, same order as gemm_bf16_t
        const uint4* w4 = (const uint4*) Ws;
        float acc = 0.f;
        for (int j = lane; j < n8; j += 32) {
            const uint4 wv = __ldg(w4 + j);
            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
            const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
            const float4 xa = *(const float4*) (xt + j * 8);
            const float4 xb = *(const float4*) (xt + j * 8 + 4);
            acc += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                 + w4f * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
        acc = dwsum(acc);
        if (lane == 0) sg1 = acc;
    }
    __syncthreads();
    slog[(size_t) t * NEXP + tid] = sp[tid];   // keep slog valid for any diagnostic reader
    if (tid == 0) sgate[t] = sg1;

    // ---- router_top10_fast_multi verbatim over the shared logits (first 8 warps) ----
    int* out_id = ids + (size_t) t * TOPK;
    float* out_w = wts + (size_t) t * TOPK;
    if (tid < 256) {
        float v0 = sp[tid], v1 = sp[tid + 256];
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

        if (warp == 0) {
            float wsum10 = 0.f;
            #pragma unroll
            for (int kk = 0; kk < TOPK; ++kk) {
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
                    out_id[kk] = bi;
                    out_w[kk] = bv;
                    sp[bi] = -2.f; // mask selected
                    wsum10 += bv;
                }
                __syncwarp();
            }
            if (lane == 0) {
                const float sc = fmaxf(wsum10, 6.103515625e-05f);
                #pragma unroll
                for (int kk = 0; kk < TOPK; ++kk) out_w[kk] /= sc;
            }
        }
    }
}

"""

# insert the fused kernel right before the routed-experts section
k = sub_once(
    k,
    "// ==================================================================\n"
    "// VECTORIZED uint32 NVFP4 ROUTED EXPERTS (multi-row T >= 1, 4x fewer load instructions!)\n",
    FUSED +
    "// ==================================================================\n"
    "// VECTORIZED uint32 NVFP4 ROUTED EXPERTS (multi-row T >= 1, 4x fewer load instructions!)\n",
    "kernel insert")

# call site: fused kernel replaces dense_bf16 + top10 ...
s = sub_once(
    s,
    "            dense_bf16(w.g_inp, mixed2.p, slog.p, NEXP, NE, T);\n"
    "            router_top10_fast_multi<<<T, 256, 0, stream>>>(slog.p, (int*) sids.p, swts.p);\n",
    "            // F1 (README 7.16): router gemm + sgate gemm + top-10 in one launch (bit-identical order)\n"
    "            router_gate_top10_fused<<<T, 512, 0, stream>>>(\n"
    "                w.g_inp.p, w.s_gate_inp.p, mixed2.p, slog.p, sgate.p, (int*) sids.p, swts.p, NE);\n",
    "[stage] fused launch")

# ... and drop the now-redundant rows=1 sgate gemm in the cold branch
s = sub_once(
    s,
    "                launch_gemm_q8p(w.sh_down.p, w.sh_down.s, xq8, xs8, sshared.p, NE, NFF, T, stream);\n"
    "                launch_gemm_bf16(w.s_gate_inp.p, mixed2.p, sgate.p, 1, NE, T, stream);\n",
    "                launch_gemm_q8p(w.sh_down.p, w.sh_down.s, xq8, xs8, sshared.p, NE, NFF, T, stream);\n",
    "[stage] sgate gemm removed")

open(K, "w", encoding="utf-8").write(k)
open(S, "w", encoding="utf-8").write(s)
print("F1 applied")
