#!/usr/bin/env python3
"""Stage A, take 5: device epoch whose VALUE is bit-identical to the old cumulative `cold_seq`.

Take 4 was deterministic but still diverged from the baseline, and the only remaining difference from
the baseline numbering was *when* the counter advances: the old code advances `cold_seq` by one per
layer only in windows that actually take the cold branch (use_cpu_pool), so prefill windows leave it
alone. Take 4 advanced once per record_forward call, shifting every decode id by the number of prefill
windows.

This version removes the difference:
  * kernel side: seq = *d_epoch + (l - l_begin) + 1, i.e. d_epoch holds the window's *base*;
  * the bump kernel adds `nlay` to d_epoch, enqueued at the END of the window in the same stream (so a
    stale kernel cannot observe the next window's value -- that was take 3's race), and only when the
    window took the cold branch;
  * host side: `seq_base` mirrors d_epoch exactly; every existing formula
    (`service_one_cold_layer`, the in-loop drain, `service_cold_layers`, and the engine's
    `stages[i].cold_seq` reads) keeps working unchanged with the same numbers as before.

Result: handshake ids are numerically identical to the baseline, and no scalar is baked into the
kernel arguments, which is the precondition for CUDA-graph capture.
"""
import shutil

ROOT = "/home/liaoyang/wzy-flash-serve/src/"
MOE, STG, ENG = ROOT + "wzy_cpu_moe.cuh", ROOT + "wzy_stage_v2.cuh", ROOT + "wzy_engine_v2.cu"

for p in (MOE, STG, ENG):   # restore the pristine baseline (saved before take 3)
    shutil.copy2("/tmp/%s.bak-epoch2" % p.rsplit("/", 1)[-1], p)


def load(p):
    s = open(p, encoding="utf-8", errors="surrogateescape", newline="").read()
    shutil.copy2(p, "/tmp/%s.bak-epoch3" % p.rsplit("/", 1)[-1])
    return s.replace("\r\n", "\n")


def sub_once(s, old, new, what):
    n = s.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:160])
    return s.replace(old, new)


# ------------------------------------------------------------------ kernels
m = load(MOE)
m = sub_once(
    m,
    "    ColdLayerReq* __restrict__ req, uint32_t seq_id,\n",
    "    ColdLayerReq* __restrict__ req, const uint32_t* __restrict__ d_epoch, int lay_off,\n",
    "split kernel signature")
m = sub_once(
    m,
    "    ColdLayerReq* __restrict__ req, const uint32_t* __restrict__ d_epoch, int lay_off,\n"
    "    int static_cap = 0, uint8_t* __restrict__ d_adapt_ref = nullptr, int* __restrict__ d_ncold_out = nullptr) {\n",
    "    ColdLayerReq* __restrict__ req, const uint32_t* __restrict__ d_epoch, int lay_off,\n"
    "    int static_cap = 0, uint8_t* __restrict__ d_adapt_ref = nullptr, int* __restrict__ d_ncold_out = nullptr) {\n"
    "    // Stage A (README 7.12): handshake id from the device epoch (the window base) instead of a baked\n"
    "    // scalar, so the window is capturable as a CUDA graph. The epoch is bumped in-stream at the\n"
    "    // window end, so this value is constant for every kernel of the window.\n"
    "    const uint32_t seq_id = (*d_epoch) + (uint32_t) lay_off + 1u;\n",
    "split kernel body")

m = sub_once(
    m,
    "const volatile ColdLayerReq* __restrict__ req, uint32_t seq_id, int n,\n",
    "const volatile ColdLayerReq* __restrict__ req, const uint32_t* __restrict__ d_epoch,\n"
    "                                             int lay_off, int n,\n",
    "wait kernel signature")
m = sub_once(
    m,
    "                                             int lay_off, int n,\n"
    "                                             const int* __restrict__ d_ncold = nullptr) {\n",
    "                                             int lay_off, int n,\n"
    "                                             const int* __restrict__ d_ncold = nullptr) {\n"
    "    const uint32_t seq_id = (*d_epoch) + (uint32_t) lay_off + 1u;   // Stage A: identical to old `seq`\n",
    "wait kernel body")

m = sub_once(
    m,
    "__global__ void split_route_hit_cold_kernel(\n",
    "// Stage A (README 7.12): advance the handshake epoch by one window's worth of layer ids. Enqueued at\n"
    "// the END of the window in the same stream, so every handshake kernel of the window has already read\n"
    "// the old base (a host-side write at window entry was racy -- take 3 of this experiment).\n"
    "__global__ void bump_epoch_kernel(unsigned int* epoch, unsigned int step) { *epoch += step; }\n\n"
    "__global__ void split_route_hit_cold_kernel(\n",
    "bump kernel")
open(MOE, "w", encoding="utf-8", errors="surrogateescape").write(m)

# ------------------------------------------------------------------ stage
s = load(STG)
s = sub_once(
    s,
    "    uint32_t cold_seq = 0;\n",
    "    uint32_t cold_seq = 0;\n"
    "    // Stage A (README 7.12): cold-handshake epoch = this window's id base. `d_epoch` is device memory,\n"
    "    // bumped in-stream at each window end by bump_epoch_kernel, so a kernel can never observe the next\n"
    "    // window's base. `seq_base` is the host mirror (no GPU read needed) and tracks the old cumulative\n"
    "    // `cold_seq` value exactly. No `seq` scalar is baked into the kernel arguments any more, which is\n"
    "    // what lets the window be captured as a CUDA graph (the bump kernel is captured with it).\n"
    "    unsigned int* d_epoch = nullptr;\n"
    "    uint32_t seq_base = 0;\n",
    "StageGpu epoch members")

# the old wrap guard keyed on the cumulative counter: recycle ids by resetting the epoch instead
s = sub_once(
    s,
    "        if (cold_seq > 0xF0000000u) {\n"
    "            CHECK(cudaStreamSynchronize(stream));\n"
    "            for (int l = l_begin; l < l_end; ++l) {\n"
    "                h_cold_req[l].ready_seq = 0u;\n"
    "                h_cold_req[l].cpu_done_seq = 0u;\n"
    "            }\n"
    "            _mm_sfence();\n"
    "            cold_seq = 0u;\n"
    "        }\n"
    "        const uint32_t base_seq = cold_seq;\n",
    "        if (seq_base > 0xF0000000u) {   // Stage A: recycle handshake ids (never reached in practice)\n"
    "            CHECK(cudaStreamSynchronize(stream));\n"
    "            for (int l = l_begin; l < l_end; ++l) {\n"
    "                h_cold_req[l].ready_seq = 0u;\n"
    "                h_cold_req[l].cpu_done_seq = 0u;\n"
    "            }\n"
    "            _mm_sfence();\n"
    "            seq_base = 0u;\n"
    "            cold_seq = 0u;\n"
    "            if (d_epoch != nullptr) CHECK(cudaMemsetAsync(d_epoch, 0, 4, stream));\n"
    "        }\n"
    "        const uint32_t base_seq = seq_base;   // Stage A: the value the kernels read from *d_epoch\n",
    "wrap guard + epoch-derived base")

s = sub_once(
    s,
    "                const uint32_t seq = ++cold_seq;\n"
    "                split_route_hit_cold_kernel<<<1, 512, 0, stream>>>(\n"
    "                    (const int*) sids.p, swts.p, T, d_lut, l, dslots, res_swts.p,\n"
    "                    grp_slot, grp_cnt, grp_e, d_ngroups, eq16, es16, d_cold_req + l, seq, static_cap, d_adapt_ref, mkey + l);",
    "                split_route_hit_cold_kernel<<<1, 512, 0, stream>>>(\n"
    "                    (const int*) sids.p, swts.p, T, d_lut, l, dslots, res_swts.p,\n"
    "                    grp_slot, grp_cnt, grp_e, d_ngroups, eq16, es16, d_cold_req + l,\n"
    "                    d_epoch, l - l_begin, static_cap, d_adapt_ref, mkey + l);",
    "split_route launch site")

s = sub_once(
    s,
    "                    comb.p, d_cold_out[l], d_cold_req + l, seq, T * NE, mkey + l);",
    "                    comb.p, d_cold_out[l], d_cold_req + l, d_epoch, l - l_begin, T * NE, mkey + l);",
    "wait_and_add launch site")

# advance the epoch for the NEXT window, in-stream, only when this window used the cold handshake
# (exactly the old `++cold_seq` condition: one increment per layer, cold-branch windows only)
s = sub_once(
    s,
    "        if (pending_bo) {\n"
    "            gr_write_multi<<<dim3(80, T), 256, 0, stream>>>(R.p, pending_bo, pending_inj, NE, HC);\n"
    "            pending_bo = nullptr; pending_inj = nullptr;\n"
    "        }",
    "        // Stage A: arm the NEXT window's handshake epoch. Stream-ordered after every handshake kernel of\n"
    "        // this window, so their ids are already fixed; the value advances exactly like the old\n"
    "        // per-layer `++cold_seq` (one window's worth of layer ids, cold-branch windows only).\n"
    "        if (use_cpu_pool) {\n"
    "            const unsigned int step = (unsigned int) (l_end - l_begin);\n"
    "            if (d_epoch != nullptr) bump_epoch_kernel<<<1, 1, 0, stream>>>(d_epoch, step);\n"
    "            seq_base += step;\n"
    "            cold_seq = seq_base;   // the base the engine reads for the NEXT window\n"
    "        }\n"
    "\n"
    "        if (pending_bo) {\n"
    "            gr_write_multi<<<dim3(80, T), 256, 0, stream>>>(R.p, pending_bo, pending_inj, NE, HC);\n"
    "            pending_bo = nullptr; pending_inj = nullptr;\n"
    "        }",
    "end-of-window epoch advance")
open(STG, "w", encoding="utf-8", errors="surrogateescape").write(s)

# ------------------------------------------------------------------ engine: device epoch per stage
e = load(ENG)
e = sub_once(
    e,
    "    CHECK(cudaMalloc(&s.dpos, 4));",
    "    // Stage A: per-stage handshake epoch (device memory; bumped in-stream at each window end).\n"
    "    CHECK(cudaMalloc((void**) &s.d_epoch, 4));\n"
    "    CHECK(cudaMemset(s.d_epoch, 0, 4));\n"
    "    CHECK(cudaMalloc(&s.dpos, 4));",
    "engine epoch alloc")
open(ENG, "w", encoding="utf-8", errors="surrogateescape").write(e)
print("stage A take5 applied: epoch value == old cumulative cold_seq")
