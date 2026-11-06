#!/usr/bin/env python3
"""Graph retry fix + memory diagnostics (README 7.15).

Fix: the `ge.second == nullptr -> break` escaped only the lookup loop, then BeginCapture ran
anyway -- so every window retried the (failing) capture. Gate the capture on a shape_dead flag.

Diagnosis for the instantiate OOM (806 nodes should need ~KBs-MB, not >64 MiB): print the real
free VRAM beside the failure so the next run tells us whether the request is huge or the heap is
fragmented.
"""
import shutil

STG = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"

s = open(STG, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(STG, "/tmp/wzy_stage_v2.cuh.bak-graph4")
s = s.replace("\r\n", "\n")


def sub_once(old, new, what):
    global s
    n = s.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:200])
    s = s.replace(old, new)


sub_once(
    "        unsigned long long gk = 0;\n"
    "        bool capturing = false;\n"
    "        if (graph_shape) {\n",
    "        unsigned long long gk = 0;\n"
    "        bool capturing = false, shape_dead = false;\n"
    "        if (graph_shape) {\n",
    "shape_dead decl")

sub_once(
    "                if (ge.second == nullptr) break;   // instantiate failed earlier: plain enqueue, no retry\n",
    "                if (ge.second == nullptr) { shape_dead = true; break; }   // no retry after an OOM\n",
    "shape_dead set")

sub_once(
    "            // First window of this shape: capture this call's enqueue sequence, then launch it.\n"
    "            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));\n"
    "            capturing = true;\n"
    "        }\n",
    "            // First window of this shape: capture this call's enqueue sequence, then launch it.\n"
    "            if (shape_dead) {\n"
    "                // plain enqueue (and fix up the window-start bookkeeping the replay path skipped)\n"
    "                if (!use_cpu_pool) CHECK(cudaMemsetAsync(dmisses, 0, 4, stream));\n"
    "            } else {\n"
    "                CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));\n"
    "                capturing = true;\n"
    "            }\n"
    "        }\n",
    "gate capture")

sub_once(
    "                std::fprintf(stderr, \"[GRAPH] shape=%llx nodes=%zu instantiate FAILED (%s) -- enqueue fallback%c\",\n"
    "                             (unsigned long long) gk, nn, cudaGetErrorString(ie), (char) 10);\n",
    "                size_t fr = 0, tt = 0;\n"
    "                cudaMemGetInfo(&fr, &tt);\n"
    "                std::fprintf(stderr, \"[GRAPH] shape=%llx nodes=%zu instantiate FAILED (%s) free=%.1f MiB -- enqueue fallback%c\",\n"
    "                             (unsigned long long) gk, nn, cudaGetErrorString(ie), (double) fr / 1048576.0, (char) 10);\n",
    "free-mem print")
open(STG, "w", encoding="utf-8").write(s)
print("graph retry fix + diag applied")
