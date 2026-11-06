#!/usr/bin/env python3
"""Repeatable equivalence probe for stage A (README 7.11).

Runs bin/wzy_pp2_engine N times with a fixed greedy prompt, records the exact token-id stream,
the DONE json and the DEC-PROF split, then compares against a saved reference file.

  python3 /tmp/eqrun.py --tag base --times 2 --save /tmp/eq_base.json
  python3 /tmp/eqrun.py --tag epoch --times 2 --ref /tmp/eq_base.json
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading

ROOT = "/home/liaoyang/wzy-flash-serve"
sys.path.insert(0, os.path.join(ROOT, "serve"))
import tokenizer as tk  # noqa: E402

PROSE = "用三句话解释为什么 MoE 推理是带宽受限的，并给出两个可落地的优化方向。"
CODE = ("写一个 C++ 环形队列，支持单生产者单消费者，要求无锁并且内存序正确。"
        "请给出完整代码，包含头文件、类定义和注释。")

PROF_RE = re.compile(
    r"\[DEC-PROF gen=(\d+) wins=(\d+) \(([\d.]+) tok/win\)\] total=([\d.]+)ms "
    r"\(([\d.]+) ms/win\) \| enq=([\d.]+)ms s0=([\d.]+)ms s1=([\d.]+)ms mtp=([\d.]+)ms "
    r"\(per win: enq=([\d.]+) s0=([\d.]+) s1=([\d.]+) mtp=([\d.]+)[^)]*\) \| hit=([\d.]+)%")


def run_one(bin_path, ids, max_tokens, env):
    proc = subprocess.Popen([bin_path], cwd=ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    err_lines = []

    def drain():
        for line in proc.stderr:
            err_lines.append(line.rstrip("\n"))

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine exited before READY:\n" + "\n".join(err_lines[-20:]))
        if line.strip().startswith("READY "):
            break
    req = {"ids": ids, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 0.0,
           "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "spec": 7, "spec_min_p": 0.72,
           "stop": [tk.ID_IM_END, tk.ID_ENDOFTEXT]}
    proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
    proc.stdin.flush()
    done, toks = None, []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine EOF before DONE:\n" + "\n".join(err_lines[-20:]))
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
    t.join(timeout=3)
    prof = None
    for ln in err_lines:
        m = PROF_RE.search(ln)
        if m:
            prof = {"tok_per_win": float(m.group(3)), "ms_per_win": float(m.group(5)),
                    "enq": float(m.group(10)), "s0": float(m.group(11)), "s1": float(m.group(12)),
                    "mtp": float(m.group(13)), "hit": float(m.group(14)), "raw": ln.strip()}
    return done, prof, toks


def tok_hash(toks):
    return hashlib.sha1(",".join(map(str, toks)).encode()).hexdigest()[:16]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "bin", "wzy_pp2_engine"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--times", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--kind", choices=["prose", "code"], default="prose")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--env", action="append", default=[],
                    help="K=V env override applied AFTER config.json env (repeatable)")
    a = ap.parse_args()

    prompt = CODE if a.kind == "code" else PROSE
    ids = tk.encode(prompt)
    bin_path = a.bin
    env = os.environ.copy()
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    for k, v in (cfg.get("env") or {}).items():
        env[k] = str(v)
    env["WZY_DEC_PROF"] = "1"
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    if a.env:
        print("env overrides:", a.env, flush=True)
    ref = json.load(open(a.ref, encoding="utf-8")) if a.ref else None
    out = []
    print("tag=%s prompt=%s(%d tok) max_new=%d ref=%s" % (a.tag, a.kind, len(ids), a.tokens,
                                                          a.ref or "-"), flush=True)
    for i in range(a.times):
        done, prof, toks = run_one(bin_path, ids, a.tokens, env)
        rec = {"run": i, "done": done, "prof": {k: v for k, v in (prof or {}).items() if k != "raw"},
               "prof_raw": (prof or {}).get("raw"), "toks": toks, "hash": tok_hash(toks)}
        out.append(rec)
        line = "  run%d: gen=%s finish=%s %.2f tok/s hash=%s" % (
            i, done.get("generated"), done.get("finish"), float(done.get("decode_tps", 0)), rec["hash"])
        if prof:
            line += " | %.2f tok/win %.1f ms/win enq=%.2f s0=%.2f s1=%.2f mtp=%.2f hit=%.1f%% promo=%s" % (
                prof["tok_per_win"], prof["ms_per_win"], prof["enq"], prof["s0"], prof["s1"],
                prof["mtp"], prof["hit"], done.get("promoted"))
        print(line, flush=True)
        if ref:
            d = first_diff(ref["toks"], toks)
            same_tokens = d == -1
            same_text = False
            if not same_tokens:
                same_text = tk.decode(ref["toks"]) == tk.decode(toks)
            print("      vs ref[:-1]: %s (first tok diff @%d%s)" % (
                "IDENTICAL" if same_tokens else ("SAME-TEXT" if same_text else "DIVERGES"),
                d, "" if same_tokens else ", ref=%s got=%s" % (
                    ref["toks"][max(0, d - 2):d + 3], toks[max(0, d - 2):d + 3])), flush=True)
        if i > 0:
            d = first_diff(out[0]["toks"], toks)
            print("      vs run0: %s (first tok diff @%d)" % (
                "IDENTICAL" if d == -1 else "DIFFERENT", d), flush=True)
    if a.save:
        json.dump(out[0], open(a.save, "w"), indent=1)
        print("saved %s" % a.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
