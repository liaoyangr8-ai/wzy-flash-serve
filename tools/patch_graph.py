#!/usr/bin/env python3
"""WZY_GRAPH_STAGE0 v2 (README 7.15): replay stage0's decode-window enqueue as a CUDA graph.

What changed since the failed 7.8 attempt:
  * The handshake seq is no longer a scalar baked into kernel args: the epoch lives in device memory
    and `bump_epoch_kernel` is enqueued at the END of the window, in-stream -- so it is captured WITH
    the window and a replay re-arms itself. No counter rewind, no `cudaStreamSynchronize` before
    replay (the two things that cost 11 ms in 7.8).
  * Promotion is deterministic now, so the equivalence criterion is strict: the token stream must be
    byte-identical to the non-graph build (same hash), not "similar".

Mechanics:
  * Key = (T, run_head, want_top16, save_snaps). First window of a shape: capture record_forward's
    enqueue sequence (cudaStreamBeginCapture ThreadLocal), EndCapture, instantiate, graphLaunch --
    so even the capture window executes.
  * Later windows of the same shape: replicate the host-side bookkeeping (pending_promo.clear,
    seq_base/cold_seq advance), copy `toks` into a stage-owned pinned buffer, then ONE
    cudaGraphLaunch. All device pointers in the graph are setup-time stable; the per-window host
    inputs (h_pos/h_emb/h_ple) are stable cudaHostAlloc buffers the engine refills before the call,
    and d_toks is fed from the pinned h_toks_g copy.
  * Replay path returns early; the engine then services stage0 exactly as in enqueue-first mode.

Enabled only when WZY_GRAPH_STAGE0 is set. Stage0 only (stage1 enqueue already runs on its own core
and is hidden under stage0 service).
"""
import shutil

STG = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"
ENG = "/home/liaoyang/wzy-flash-serve/src/wzy_engine_v2.cu"


def load(p):
    s = open(p, encoding="utf-8", errors="surrogateescape", newline="").read()
    shutil.copy2(p, "/tmp/%s.bak-graph2" % p.rsplit("/", 1)[-1])
    return s.replace("\r\n", "\n")


def sub_once(s, old, new, what, where=""):
    n = s.count(old)
    assert n == 1, "%s%s: anchor count %d (expected 1)\n%r" % (where, what, n, old[:200])
    return s.replace(old, new)


# ------------------------------------------------------------------ stage members
s = load(STG)
s = sub_once(
    s,
    "    unsigned int* d_epoch = nullptr;\n"
    "    uint32_t seq_base = 0;\n",
    "    unsigned int* d_epoch = nullptr;\n"
    "    uint32_t seq_base = 0;\n"
    "    // CUDA graph replay (WZY_GRAPH_STAGE0, README 7.15): one exec per window shape\n"
    "    // (T, run_head, want_top16, save_snaps); decode only ever produces a handful of shapes.\n"
    "    std::vector<std::pair<unsigned long long, cudaGraphExec_t>> graph_execs;\n"
    "    uint32_t* h_toks_g = nullptr;   // pinned, stage-owned: the captured d_toks copy reads this\n",
    "graph members", "[stage] ")

# ------------------------------------------------------------------ record_forward: replay/capture switch
s = sub_once(
    s,
    "        if (seq_base > 0xF0000000u) {   // Stage A: recycle handshake ids (never reached in practice)\n",
    "        // ---- CUDA graph replay (WZY_GRAPH_STAGE0, README 7.15) -------------------------------\n"
    "        static const bool graph_on = std::getenv(\"WZY_GRAPH_STAGE0\") != nullptr;\n"
    "        const bool graph_shape = graph_on && use_cpu_pool && toks != nullptr && h_toks_g != nullptr;\n"
    "        unsigned long long gk = 0;\n"
    "        bool capturing = false;\n"
    "        if (graph_shape) {\n"
    "            std::memcpy(h_toks_g, toks, (size_t) T * 4);   // captured H2D copy reads this pinned buffer\n"
    "            gk = ((unsigned long long) T << 8) | (run_head ? 4ull : 0ull) | (want_top16 ? 2ull : 0ull)\n"
    "               | (save_snaps ? 1ull : 0ull);\n"
    "            for (const auto& ge : graph_execs) {\n"
    "                if (ge.first != gk) continue;\n"
    "                // Replay: host bookkeeping only, then one launch. The epoch bump is INSIDE the capture,\n"
    "                // so the next window's handshake ids are armed without any host write or stream sync.\n"
    "                pending_promo.clear();\n"
    "                seq_base += (uint32_t) (l_end - l_begin);\n"
    "                cold_seq = seq_base;\n"
    "                CHECK(cudaGraphLaunch(ge.second, stream));\n"
    "                return;\n"
    "            }\n"
    "            // First window of this shape: capture this call's enqueue sequence, then launch it.\n"
    "            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));\n"
    "            capturing = true;\n"
    "        }\n"
    "        if (seq_base > 0xF0000000u) {   // Stage A: recycle handshake ids (never reached in practice)\n",
    "graph replay/capture switch", "[stage] ")

# ------------------------------------------------------------------ d_toks memcpy from the pinned buffer
s = sub_once(
    s,
    "                CHECK(cudaMemcpyAsync(d_toks, toks, (size_t) T * 4, cudaMemcpyHostToDevice, stream));\n",
    "                CHECK(cudaMemcpyAsync(d_toks, graph_shape ? (const int*) h_toks_g : toks,\n"
    "                                      (size_t) T * 4, cudaMemcpyHostToDevice, stream));\n",
    "d_toks memcpy source", "[stage] ")

# ------------------------------------------------------------------ close the capture at end of body
s = sub_once(
    s,
    "        if (inline_service && use_cpu_pool && l_end > l_begin) {\n"
    "            if (cold_inline) {\n"
    "                service_one_cold_layer(l_end - 1, host_pack, cpu_pool, base_seq, T, false);\n"
    "            } else {\n"
    "                // Enqueue-first mode: everything after the first layer is still pending, so drain the\n"
    "                // whole stage in layer order (the GPU is spinning on the earliest unserved layer).\n"
    "                for (int l = l_begin; l < l_end; ++l)\n"
    "                    service_one_cold_layer(l, host_pack, cpu_pool, base_seq, T, false);\n"
    "            }\n"
    "        }\n"
    "    }\n",
    "        if (inline_service && use_cpu_pool && l_end > l_begin) {\n"
    "            if (cold_inline) {\n"
    "                service_one_cold_layer(l_end - 1, host_pack, cpu_pool, base_seq, T, false);\n"
    "            } else {\n"
    "                // Enqueue-first mode: everything after the first layer is still pending, so drain the\n"
    "                // whole stage in layer order (the GPU is spinning on the earliest unserved layer).\n"
    "                for (int l = l_begin; l < l_end; ++l)\n"
    "                    service_one_cold_layer(l, host_pack, cpu_pool, base_seq, T, false);\n"
    "            }\n"
    "        }\n"
    "\n"
    "        if (capturing) {\n"
    "            cudaGraph_t g = nullptr;\n"
    "            CHECK(cudaStreamEndCapture(stream, &g));\n"
    "            cudaGraphExec_t ex = nullptr;\n"
    "            CHECK(cudaGraphInstantiate(&ex, g, 0));\n"
    "            graph_execs.push_back({gk, ex});\n"
    "            cudaGraphDestroy(g);\n"
    "            CHECK(cudaGraphLaunch(ex, stream));   // the capture window must still execute\n"
    "        }\n"
    "    }\n",
    "capture close", "[stage] ")
open(STG, "w", encoding="utf-8", errors="surrogateescape").write(s)

# ------------------------------------------------------------------ engine: pinned toks buffer per stage
e = load(ENG)
e = sub_once(
    e,
    "    CHECK(cudaMalloc(&s.dpos, 4));",
    "    CHECK(cudaHostAlloc((void**) &s.h_toks_g, 64 * 4, cudaHostAllocMapped | cudaHostAllocPortable));\n"
    "    CHECK(cudaMalloc(&s.dpos, 4));",
    "pinned toks buffer", "[engine] ")
open(ENG, "w", encoding="utf-8", errors="surrogateescape").write(e)
print("WZY_GRAPH_STAGE0 v2 applied")
