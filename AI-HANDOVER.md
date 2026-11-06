# AI handover - WZY

This is a source snapshot renamed from the former project name to WZY, including environment variables, identifiers, filenames and UI text. The renamed code has not been built or tested. No engine binary or model weights are included. Start by reading README.md and DEPLOYMENT-SOP.md; inspect the target Linux machine, model paths and CUDA dependencies. Run ./build.sh and test text, protocol and vision before deployment. Original environment: two 22GB RTX 2080 Ti, AVX2/FMA CPU and approximately 94 GiB RAM. Vision also requires a compatible strata-vision executable and mmproj model. Config is a template; set correct paths and secure API authentication before exposing the service.

---

## Latest research state (2026-11-06)

**Read `PROJECT-SUMMARY.md` first.** It is the one-page handover; README sections 7.12-7.20 are the
full record with every intermediate state, including the negative results.

Headline numbers on the reference machine (2x RTX 2080 Ti, 2000-token Chinese story prompt, greedy):

| milestone | value |
| --- | --- |
| decode throughput, before this round | 62.1 tok/s |
| decode throughput, current | **65.7 tok/s** (+5.8%) |
| greedy decode reproducibility | **bit-identical across runs** (was 2-3 different streams) |

What landed in the source (all verified byte-identical against the previous token stream, except
where noted):

1. **Deterministic promotion order** (`src/wzy_stage_v2.cuh`, `tools/patch_promo_order.py`) - the root
   cause of irreproducible greedy decode: cold groups are handed out by `atomicAdd`, the service loop
   pushed promotion candidates in that order and truncated at `WZY_PROMO_QUEUE`, so the promoted
   expert subset changed every run. Now every eligible candidate is collected and
   `promote_cold` sorts by `(freq desc, gkey asc)` before applying the cap.
2. **Stage A epoch handshake** (`tools/patch_stageA_epoch.py`, already applied): the kernels no longer
   take the handshake `seq` as a scalar argument but read a device epoch that is bumped in-stream at
   the end of each window, which is what makes a window capturable as a CUDA graph.
3. **CUDA graph capture/replay** (`WZY_GRAPH_STAGE0=1`, off by default, with `WZY_GRAPH_MAXW`):
   works and is byte-equivalent, but measured *burst-neutral* - the enqueue time it removes is time
   the CPU was spending servicing cold experts in parallel, so `enq + s0` stays ~14.5 ms/window. Do
   not expect a speedup from the graph alone; README 7.15 explains why.
4. **F1b** (`tools/patch_f1b.py`): the rows=1 shared-expert gate GEMM is folded into the idle warp 1
   of `router_top10_fast_multi`. Bit-identical, measured neutral.
5. **Production tuning**: `WZY_PROMO_MIN_F=2 / WZY_PROMO_PER_WIN=16 / WZY_PROMO_QUEUE=64` in
   `config.json`, plus the strata-official CJK draft vocabulary and the learned `expert-profile`
   (+1.2 tok/s from the profile alone, measured in a controlled A/B - README 7.20).

Where the remaining time goes (README 7.16, `WZY_LAYER_PROF=1`): moe 0.252 ms/layer, attention
0.176, router 0.060, gr_read 0.047 - GPU-bound at ~20 us per kernel, so the next lever is *inside*
kernels (intermediate-buffer traffic, the drafter's 114452-row head GEMM, the ring-attention shared
memory scan), not kernel-boundary fusion. Two fusion attempts are recorded as negative results
(README 7.17, 7.18) with the methodology correction they produced: fusing is only worth it if it
keeps the total warp count and reduces GPU serial work.

Tooling shipped with this snapshot (see `tools/`): `eqrun.py` (byte-level equivalence harness with
env overrides and DEC-PROF capture), `crab.py` (fixed long-form Chinese A/B), and the patch scripts
above; `bench/bench_decode.py` remains the config sweep.
