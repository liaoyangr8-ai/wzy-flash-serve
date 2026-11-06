#!/usr/bin/env python3
"""Decode-speed A/B sweep for the single-stream engine (wzy_pp2_engine).

The engine reports `decode_tps` in its DONE line and, with WZY_DEC_PROF=1, a `[DEC-PROF ...]` line on
stderr that decomposes each verify window into enqueue / stage-0 cold / stage-1 cold / drafter. That
split is what decides which knob matters, so this script prints both.

Each config is a separate process (env is per-process), so every row pays the full boot: ~60-90 s for
pinning the 63.3 GiB expert pack. Budget ~10 minutes for the default sweep.

Usage:
  python3 bench/bench_decode.py                      # default sweep, prose prompt
  python3 bench/bench_decode.py --kind code          # structured-output prompt
  python3 bench/bench_decode.py --only 2,4           # run a subset
  python3 bench/bench_decode.py --tokens 160
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "serve"))
import tokenizer as tk  # noqa: E402

PROSE = "用三句话解释为什么 MoE 推理是带宽受限的，并给出两个可落地的优化方向。"
CODE = ("写一个 C++ 环形队列，支持单生产者单消费者，要求无锁并且内存序正确。"
        "请给出完整代码，包含头文件、类定义和注释。")

REF_LABEL = "NO MTP (pure argmax reference)"

# (label, env overrides). Every row inherits config.json's env, then applies these.
SWEEP = [
    (REF_LABEL, {"WZY_NOMTP": "1"}),
    ("depth=2", {}),
    ("depth=7", {"WZY_SPEC_DEPTH": "7"}),
]

PROF_RE = re.compile(
    r"\[DEC-PROF gen=(\d+) wins=(\d+) \(([\d.]+) tok/win\)\] total=([\d.]+)ms "
    r"\(([\d.]+) ms/win\) \| enq=([\d.]+)ms s0=([\d.]+)ms s1=([\d.]+)ms mtp=([\d.]+)ms "
    r"\(per win: enq=([\d.]+) s0=([\d.]+) s1=([\d.]+) mtp=([\d.]+)\) \| hit=([\d.]+)%")


def run_one(bin_path, ids, max_tokens, env):
    proc = subprocess.Popen(
        [bin_path], cwd=ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    err_lines = []

    def drain():
        for line in proc.stderr:
            err_lines.append(line.rstrip("\n"))

    t_err = threading.Thread(target=drain, daemon=True)
    t_err.start()

    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine exited before READY:\n" + "\n".join(err_lines[-30:]))
        if line.strip().startswith("READY "):
            break

    req = {
        "ids": ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,          # greedy: the reproducible baseline
        "top_p": 0.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "spec": 7,
        "spec_min_p": 0.72,
        "stop": [tk.ID_IM_END, tk.ID_ENDOFTEXT],
    }
    proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
    proc.stdin.flush()

    done = None
    toks = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine EOF before DONE:\n" + "\n".join(err_lines[-30:]))
        s = line.strip()
        if s.startswith("TOK "):
            toks.append(int(s.split()[-1]))
        elif s.startswith("DONE "):
            done = json.loads(s[5:])
            break

    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    t_err.join(timeout=3)

    prof = None
    for ln in err_lines:
        m = PROF_RE.search(ln)
        if m:
            prof = {
                "tok_per_win": float(m.group(3)),
                "ms_per_win": float(m.group(5)),
                "enq": float(m.group(10)),
                "s0": float(m.group(11)),
                "s1": float(m.group(12)),
                "mtp": float(m.group(13)),
                "hit": float(m.group(14)),
            }
    return done, prof, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["prose", "code"], default="prose")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--only", default="")
    ap.add_argument("--ctx", default=None, help="override WZY_CTX (default: config.json)")
    args = ap.parse_args()

    prompt = CODE if args.kind == "code" else PROSE
    ids = tk.encode(prompt)
    bin_path = os.path.join(ROOT, "bin", "wzy_pp2_engine")
    if not os.path.exists(bin_path):
        print("missing %s (run ./build.sh)" % bin_path)
        return 2

    base_env = os.environ.copy()
    try:
        cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
        for k, v in (cfg.get("env") or {}).items():
            base_env[k] = str(v)
    except Exception:
        pass
    base_env["WZY_DEC_PROF"] = "1"
    if args.ctx:
        base_env["WZY_CTX"] = str(args.ctx)

    picks = [int(x) for x in args.only.split(",") if x.strip()] if args.only else list(range(len(SWEEP)))
    rows = []
    print("prompt=%s (%d tokens), max_new=%d, ctx=%s\n" % (args.kind, len(ids), args.tokens, base_env.get("WZY_CTX")))
    for i in picks:
        label, over = SWEEP[i]
        env = dict(base_env)
        env.update(over)
        print("[%d/%d] %s ..." % (i + 1, len(SWEEP), label), flush=True)
        try:
            done, prof, toks = run_one(bin_path, ids, args.tokens, env)
        except Exception as exc:
            print("    FAILED: %s" % str(exc).splitlines()[0])
            continue
        rows.append((label, done, prof, toks))
        line = "    %d tok, finish=%s, %.2f tok/s" % (
            int(done.get("generated", 0)), done.get("finish"), float(done.get("decode_tps", 0.0)))
        if prof:
            line += " | %.2f tok/win, %.1f ms/win | enq=%.2f s0=%.2f s1=%.2f mtp=%.2f | hit=%.1f%%" % (
                prof["tok_per_win"], prof["ms_per_win"], prof["enq"], prof["s0"], prof["s1"], prof["mtp"], prof["hit"])
        print(line, flush=True)

    if not rows:
        return 1

    # The ONLY correct reference for a lossless-greedy check is pure autoregressive argmax
    # (WZY_NOMTP=1). Comparing rows against "whichever row ran first" lets a scheduling knob
    # become the baseline, which is how earlier runs produced false DIVERGES verdicts.
    ref = next((r for r in rows if r[0] == REF_LABEL), None)
    if ref is None:
        print("\nWARNING: reference row '%s' was not run; falling back to the first row." % REF_LABEL)
        ref = rows[0]
    _, _, _, ref_toks = ref
    ref_text = tk.decode(ref_toks)
    print("\nreference = %s: %d tokens / %d chars" % (ref[0], len(ref_toks), len(ref_text)))

    def verdict_of(toks):
        """identical | same-text | DIVERGES. Token ids can differ while the text is identical
        (two BPE splits of the same word), so a token-id-only check reports false regressions."""
        if toks == ref_toks:
            return "identical", True
        text = tk.decode(toks)
        if text == ref_text:
            return "same-text", True
        at = next((k for k, (a, b) in enumerate(zip(ref_toks, toks)) if a != b), min(len(ref_toks), len(toks)))
        ct = next((k for k, (a, b) in enumerate(zip(ref_text, text)) if a != b), min(len(ref_text), len(text)))
        return "DIVERGES@tok%d,char%d" % (at, ct), False

    print("%-30s %9s %9s %9s %8s %8s %8s %8s %s" % (
        "config", "tok/s", "tok/win", "ms/win", "enq", "s0", "s1", "mtp", "vs ref"))
    bad = []
    for label, done, prof, toks in sorted(rows, key=lambda r: -float(r[1].get("decode_tps", 0.0))):
        verdict, ok = verdict_of(toks)
        if not ok:
            bad.append(label)
        if prof:
            print("%-30s %9.2f %9.2f %9.2f %8.2f %8.2f %8.2f %8.2f %s" % (
                label, float(done.get("decode_tps", 0.0)), prof["tok_per_win"], prof["ms_per_win"],
                prof["enq"], prof["s0"], prof["s1"], prof["mtp"], verdict))
        else:
            print("%-30s %9.2f %9s %9s %8s %8s %8s %8s %s" % (
                label, float(done.get("decode_tps", 0.0)), "-", "-", "-", "-", "-", "-", verdict))

    if bad:
        print("\nREGRESSION: these configs changed the decoded TEXT: %s" % ", ".join(bad))
        print("A lossless greedy decode must not depend on scheduling knobs. Do not ship them.")
        ref_label, _, _, ref_toks = ref
        for label, _, _, toks in rows:
            if label in bad:
                at = next((k for k, (a, b) in enumerate(zip(ref_toks, toks)) if a != b), min(len(ref_toks), len(toks)))
                lo = max(0, at - 6)
                print("  %s @%d\n    ref : %r\n    this: %r" % (
                    label, at, tk.decode(ref_toks[lo:at + 6]), tk.decode(toks[lo:at + 6])))
        return 1

    best = max(rows, key=lambda r: float(r[1].get("decode_tps", 0.0)))
    print("\nbest: %s @ %.2f tok/s" % (best[0], float(best[1].get("decode_tps", 0.0))))
    print("Read the split: if mtp dominates, gate speculation harder; if s0+s1 dominates, the fix is "
          "hit-rate/cold-expert work; if neither dominates, it is GPU kernel latency and the lever is "
          "cross-window pipelining and KV ring size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
