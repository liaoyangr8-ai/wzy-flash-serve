#!/usr/bin/env python3
"""F1b (README 7.18): fold the rows=1 shared-expert gate gemm into router_top10_fast_multi.

`launch_gemm_bf16(w.s_gate_inp, mixed2, sgate, 1, NE, T)` is a full launch in which only warp 0 of
one block does any work. Inside `router_top10_fast_multi`, warp 1 is idle during the whole top-10
phase, so it computes the same dot product for free -- one launch per layer removed, bit-identical:

  * per (token) accumulation order == gemm_bf16_t rows=1: lane-strided j loop over cols/8 uint4
    pairs, the same 8-product expression, dwsum reduction (replicated exactly);
  * for T > 8 the old path used gemm_bf16_row1_batch_kernel (8 tokens/block) -- per (token) order
    is the same lane-strided loop, so the per-token result is bit-identical there too;
  * mixed2 is unchanged between the router gemm and the old sgate gemm site (only eq32/es32/eq16/
    es16 quant outputs are written in between), so computing sgate early is value-identical.

The cold-branch sgate launch and the dense-branch one are both removed; every consumer of sgate
(moe_combine_shared_multi) reads the same buffer.
"""
import shutil

K = "/home/liaoyang/wzy-flash-serve/src/wzy_kernels_v2.cuh"
S = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"

k = open(K, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(K, "/tmp/wzy_kernels_v2.cuh.bak-f1b")
k = k.replace("\r\n", "\n")

s = open(S, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(S, "/tmp/wzy_stage_v2.cuh.bak-f1b")
s = s.replace("\r\n", "\n")


def sub_once(txt, old, new, what):
    n = txt.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:200])
    return txt.replace(old, new)


# ---- 1. kernel: signature + sgate dot on warp 1 during the top-10 phase -------------------------
k = sub_once(
    k,
    "__global__ void router_top10_fast_multi(const float* __restrict__ logits, int* __restrict__ ids,\n"
    "                                        float* __restrict__ wts) {\n"
    "    const int t = blockIdx.x;\n"
    "    const float* x = logits + (size_t) t * NEXP;\n",
    "__global__ void router_top10_fast_multi(const float* __restrict__ logits, int* __restrict__ ids,\n"
    "                                        float* __restrict__ wts,\n"
    "                                        const unsigned short* __restrict__ sg_w,\n"
    "                                        const float* __restrict__ sg_x, float* __restrict__ sg_out,\n"
    "                                        int sg_cols) {\n"
    "    const int t = blockIdx.x;\n"
    "    const float* x = logits + (size_t) t * NEXP;\n",
    "kernel signature")

k = sub_once(
    k,
    "    // Warp 0 performs 10 parallel warp-wide argmax passes over the 512 probabilities in shared memory!\n"
    "    if (warp == 0) {\n",
    "    // F1b: warp 1 is idle from here on -- it computes the rows=1 shared-expert gate dot product with\n"
    "    // the exact gemm_bf16_t rows=1 accumulation order (bit-identical), saving one launch per layer.\n"
    "    if (warp == 1 && sg_w != nullptr) {\n"
    "        const uint4* w4 = (const uint4*) sg_w;\n"
    "        const float* xt = sg_x + (size_t) t * sg_cols;\n"
    "        const int n8 = sg_cols / 8;\n"
    "        float acc = 0.f;\n"
    "        for (int j = lane; j < n8; j += 32) {\n"
    "            const uint4 wv = __ldg(w4 + j);\n"
    "            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);\n"
    "            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);\n"
    "            const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);\n"
    "            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);\n"
    "            const float4 xa = *(const float4*) (xt + j * 8);\n"
    "            const float4 xb = *(const float4*) (xt + j * 8 + 4);\n"
    "            acc += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w\n"
    "                 + w4f * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;\n"
    "        }\n"
    "        acc = dwsum(acc);\n"
    "        if (lane == 0) sg_out[t] = acc;\n"
    "    }\n"
    "\n"
    "    // Warp 0 performs 10 parallel warp-wide argmax passes over the 512 probabilities in shared memory!\n"
    "    if (warp == 0) {\n",
    "warp-1 sgate")

# ---- 2. stage call sites ------------------------------------------------------------------------
s = sub_once(
    s,
    "            dense_bf16(w.g_inp, mixed2.p, slog.p, NEXP, NE, T);\n"
    "            router_top10_fast_multi<<<T, 256, 0, stream>>>(slog.p, (int*) sids.p, swts.p);\n",
    "            dense_bf16(w.g_inp, mixed2.p, slog.p, NEXP, NE, T);\n"
    "            // F1b: warp 1 of the top-10 kernel also computes the rows=1 sgate gemm (saves a launch)\n"
    "            router_top10_fast_multi<<<T, 256, 0, stream>>>(slog.p, (int*) sids.p, swts.p,\n"
    "                                                            w.s_gate_inp.p, mixed2.p, sgate.p, NE);\n",
    "[stage] cold-branch call")

s = sub_once(
    s,
    "                silu_mul_multi<<<(NFF * T + 255) / 256, 256, 0, stream>>>(sff.p, sup.p, tmp_ff.p, NFF * T);\n"
    "                dense_q8(w.sh_down, tmp_ff.p, sshared.p, NE, NFF, T);\n"
    "                launch_gemm_bf16(w.s_gate_inp.p, mixed2.p, sgate.p, 1, NE, T, stream);\n",
    "                silu_mul_multi<<<(NFF * T + 255) / 256, 256, 0, stream>>>(sff.p, sup.p, tmp_ff.p, NFF * T);\n"
    "                dense_q8(w.sh_down, tmp_ff.p, sshared.p, NE, NFF, T);\n"
    "                // F1b: sgate computed inside router_top10_fast_multi (warp 1)\n",
    "[stage] cold-branch sgate launch removed")

open(K, "w", encoding="utf-8").write(k)
open(S, "w", encoding="utf-8").write(s)
print("F1b applied")
