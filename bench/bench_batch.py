#!/usr/bin/env python3
"""Aggregate-throughput benchmark for the concurrent batch engine.

Submits N requests at once (default = the engine's slot count), counts every TOK line, and reports:
  * aggregate tok/s  (all generated tokens / wall clock between first and last token)
  * per-request tok/s
  * interleaving proof: the number of requests that had a token before any request finished

Usage: python3 bench/bench_batch.py [N] [max_tokens]
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "serve"))
import tokenizer as tk  # noqa: E402

PROMPTS = [
    "用三句话解释为什么 MoE 推理是带宽受限的，并给出两个可落地的优化方向。",
    "写一个 C++ 环形队列，支持单生产者单消费者，要求无锁并且内存序正确。",
    "解释 CUDA 中 __syncthreads 与 __threadfence 的区别，各举一个误用场景。",
    "把下面这段 Python 改成向量化实现：for i in range(n): y[i] = a*x[i] + b。",
    "分析 2080 Ti 上 INT8 KV cache 的收益与风险，给出量化误差的估计方法。",
    "总结投机解码在贪心解码下的正确性条件，说明为什么必须逐 token 精确验证。",
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    bin_path = os.path.join(ROOT, "bin", "wzy_batch_engine")
    if not os.path.exists(bin_path):
        print("missing %s (run ./build.sh)" % bin_path)
        return 2

    env = os.environ.copy()
    env.setdefault("WZY_BATCH", str(n))
    env.setdefault("WZY_CTX", "32768")

    proc = subprocess.Popen(
        [bin_path], cwd=ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    while True:
        line = proc.stdout.readline()
        if not line:
            print(proc.stderr.read()[-4000:])
            return 1
        if line.strip().startswith("READY "):
            ready = json.loads(line.strip()[6:])
            print("engine READY: slots=%s ctx=%s" % (ready.get("sessions"), ready.get("ctx_max")))
            break

    ids = [tk.encode(PROMPTS[i % len(PROMPTS)]) for i in range(n)]
    for i, seq in enumerate(ids):
        req = {
            "req": i + 1,
            "ids": seq,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "stop": [tk.ID_IM_END, tk.ID_ENDOFTEXT],
        }
        proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
    proc.stdin.flush()

    t0 = time.time()
    first_tok = {}
    count = {}
    finished = {}
    order = []
    while len(finished) < n:
        line = proc.stdout.readline()
        if not line:
            print(proc.stderr.read()[-4000:])
            break
        s = line.strip()
        now = time.time()
        if s.startswith("TOK "):
            _, rid, _tok = s.split()
            rid = int(rid)
            count[rid] = count.get(rid, 0) + 1
            if rid not in first_tok:
                first_tok[rid] = now
                order.append(rid)
        elif s.startswith("DONE "):
            dn = json.loads(s[5:])
            finished[dn.get("req")] = (now, dn)

    total = sum(count.values())
    span = max(v[0] for v in finished.values()) - min(first_tok.values())
    wall = time.time() - t0
    print("requests=%d  tokens=%d  span=%.2fs  wall=%.2fs" % (n, total, span, wall))
    print("aggregate: %.1f tok/s (over first-token -> last-DONE span)" % (total / span if span > 0 else 0.0))
    for rid in sorted(finished):
        t_end, dn = finished[rid]
        dur = t_end - first_tok.get(rid, t_end)
        per = count.get(rid, 0) / dur if dur > 0 else 0.0
        print("  req %d: %d tokens, %.1f tok/s, finish=%s" % (
            rid, count.get(rid, 0), per, dn.get("finish")))
    print("interleave order:", order)

    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
