#!/usr/bin/env python3
"""Graph shape gate (README 7.15): each 806-node exec costs ~40 MiB VRAM, and the pre-READY warmup
runs wide windows (T=8/6) whose shapes never occur in decode -- they ate the whole 148 MiB headroom
before the real decode shapes (T=2/3) could instantiate.

Gate capture to T <= WZY_GRAPH_MAXW (default 4), which covers the decoded window widths.
"""
import shutil

STG = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"

s = open(STG, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(STG, "/tmp/wzy_stage_v2.cuh.bak-graph5")
s = s.replace("\r\n", "\n")


def sub_once(old, new, what):
    global s
    n = s.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:200])
    s = s.replace(old, new)


sub_once(
    "        static const bool graph_on = std::getenv(\"WZY_GRAPH_STAGE0\") != nullptr;\n"
    "        const bool graph_shape = graph_on && use_cpu_pool && toks != nullptr && h_toks_g != nullptr;\n",
    "        static const bool graph_on = std::getenv(\"WZY_GRAPH_STAGE0\") != nullptr;\n"
    "        // Each 806-node exec costs tens of MiB of VRAM (see [GRAPH] diagnostics), and the warmup runs\n"
    "        // wide shapes that decode never produces -- gate capture to the widths decode actually hits.\n"
    "        static const int graph_maxw = std::getenv(\"WZY_GRAPH_MAXW\") ? std::atoi(std::getenv(\"WZY_GRAPH_MAXW\")) : 4;\n"
    "        const bool graph_shape = graph_on && use_cpu_pool && toks != nullptr && h_toks_g != nullptr\n"
    "                                 && T <= graph_maxw;\n",
    "maxw gate")
open(STG, "w", encoding="utf-8").write(s)
print("graph maxw gate applied")
