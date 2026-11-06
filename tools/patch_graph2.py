#!/usr/bin/env python3
"""Graph-instantiate robustness (README 7.15): node-count probe + OOM fallback.

The first v2 build died with `cudaGraphInstantiate` OOM even with 64 MiB free, which is far more
than a ~700-node exec should need -- so measure the real node count and stop making OOM fatal.

Changes:
  * capture close: EndCapture failure stays fatal (broken stream), but instantiate failure now
    prints `[GRAPH] shape=... nodes=... instantiate OOM (...) -- enqueue fallback`, records the
    shape as never-retry, and falls back: undo this call's host bookkeeping (seq_base/cold_seq were
    already advanced; the captured kernels never executed, so d_epoch was NOT bumped) and re-run
    the body once via recursion, which lands on the plain-enqueue path.
  * replay loop skips shapes recorded with a nullptr exec.
"""
import shutil

STG = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"

s = open(STG, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(STG, "/tmp/wzy_stage_v2.cuh.bak-graph3")
s = s.replace("\r\n", "\n")


def sub_once(old, new, what):
    global s
    n = s.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:200])
    s = s.replace(old, new)


# ---- 1. remember the pre-call base for the fallback; skip nullptr (failed) shapes ---------------
sub_once(
    "        // ---- CUDA graph replay (WZY_GRAPH_STAGE0, README 7.15) -------------------------------\n"
    "        static const bool graph_on = std::getenv(\"WZY_GRAPH_STAGE0\") != nullptr;\n",
    "        // ---- CUDA graph replay (WZY_GRAPH_STAGE0, README 7.15) -------------------------------\n"
    "        const uint32_t seq_base0 = seq_base;   // restored if instantiate fails (see capture close)\n"
    "        static const bool graph_on = std::getenv(\"WZY_GRAPH_STAGE0\") != nullptr;\n",
    "seq_base0")

sub_once(
    "            for (const auto& ge : graph_execs) {\n"
    "                if (ge.first != gk) continue;\n"
    "                // Replay: host bookkeeping only, then one launch. The epoch bump is INSIDE the capture,\n"
    "                // so the next window's handshake ids are armed without any host write or stream sync.\n"
    "                pending_promo.clear();\n"
    "                seq_base += (uint32_t) (l_end - l_begin);\n"
    "                cold_seq = seq_base;\n"
    "                CHECK(cudaGraphLaunch(ge.second, stream));\n"
    "                return;\n"
    "            }\n",
    "            for (const auto& ge : graph_execs) {\n"
    "                if (ge.first != gk) continue;\n"
    "                if (ge.second == nullptr) break;   // instantiate failed earlier: plain enqueue, no retry\n"
    "                // Replay: host bookkeeping only, then one launch. The epoch bump is INSIDE the capture,\n"
    "                // so the next window's handshake ids are armed without any host write or stream sync.\n"
    "                pending_promo.clear();\n"
    "                seq_base += (uint32_t) (l_end - l_begin);\n"
    "                cold_seq = seq_base;\n"
    "                CHECK(cudaGraphLaunch(ge.second, stream));\n"
    "                return;\n"
    "            }\n",
    "nullptr-shape skip")

# ---- 2. capture close: probe node count, survive instantiate OOM --------------------------------
sub_once(
    "        if (capturing) {\n"
    "            cudaGraph_t g = nullptr;\n"
    "            CHECK(cudaStreamEndCapture(stream, &g));\n"
    "            cudaGraphExec_t ex = nullptr;\n"
    "            CHECK(cudaGraphInstantiate(&ex, g, 0));\n"
    "            graph_execs.push_back({gk, ex});\n"
    "            cudaGraphDestroy(g);\n"
    "            CHECK(cudaGraphLaunch(ex, stream));   // the capture window must still execute\n"
    "        }\n",
    "        if (capturing) {\n"
    "            cudaGraph_t g = nullptr;\n"
    "            CHECK(cudaStreamEndCapture(stream, &g));\n"
    "            size_t nn = 0;\n"
    "            cudaGraphGetNodes(g, nullptr, &nn);\n"
    "            cudaGraphExec_t ex = nullptr;\n"
    "            cudaError_t ie = cudaGraphInstantiate(&ex, g, 0);\n"
    "            if (ie == cudaSuccess) {\n"
    "                graph_execs.push_back({gk, ex});\n"
    "                std::fprintf(stderr, \"[GRAPH] shape=%llx nodes=%zu instantiated%c\",\n"
    "                             (unsigned long long) gk, nn, (char) 10);\n"
    "                cudaGraphDestroy(g);\n"
    "                CHECK(cudaGraphLaunch(ex, stream));   // the capture window must still execute\n"
    "            } else {\n"
    "                graph_execs.push_back({gk, nullptr});   // never retry this shape\n"
    "                cudaGraphDestroy(g);\n"
    "                std::fprintf(stderr, \"[GRAPH] shape=%llx nodes=%zu instantiate FAILED (%s) -- enqueue fallback%c\",\n"
    "                             (unsigned long long) gk, nn, cudaGetErrorString(ie), (char) 10);\n"
    "                // The captured call never executed (d_epoch NOT bumped) but already advanced the host\n"
    "                // bookkeeping: undo it and re-run the body once, landing on the plain-enqueue path.\n"
    "                seq_base = seq_base0;\n"
    "                cold_seq = seq_base0;\n"
    "                record_forward(h_pos, h_emb, h_ple, T, run_head, save_snaps, host_pack, cpu_pool,\n"
    "                               want_top16, snap_offset, inline_service, peer_s0, peer_base, peer_l0,\n"
    "                               self_l, toks);\n"
    "                return;\n"
    "            }\n"
    "        }\n",
    "capture close v2")
open(STG, "w", encoding="utf-8").write(s)
print("graph robustness patch applied")
