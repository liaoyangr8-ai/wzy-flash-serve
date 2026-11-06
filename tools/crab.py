# -*- coding: utf-8 -*-
"""Crab-story A/B: fixed prompt, 2000 tokens, greedy; measures tok/win, hit, ms/win, tok/s
and saves the text per variant. Usage: python3 crab.py --tag A [--save /tmp/crab_A.json]"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = "/home/liaoyang/wzy-flash-serve"
sys.path.insert(0, os.path.join(ROOT, "serve"))
import tokenizer as tk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="A")
    ap.add_argument("--save", default=None)
    ap.add_argument("--tokens", type=int, default=2000)
    a = ap.parse_args()

    ids = [int(x) for x in open("/tmp/ids_crab.txt").read().split()]
    env = os.environ.copy()
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    for k, v in (cfg.get("env") or {}).items():
        env[k] = str(v)
    env["WZY_DEC_PROF"] = "1"

    proc = subprocess.Popen([os.path.join(ROOT, "bin", "wzy_pp2_engine")], cwd=ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    err = []
    threading.Thread(target=lambda: [err.append(l.rstrip()) for l in proc.stderr], daemon=True).start()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("no READY: " + "\n".join(err[-5:]))
        if line.strip().startswith("READY "):
            break
    req = {"ids": ids, "max_tokens": a.tokens, "temperature": 0.0, "top_p": 0.0, "top_k": 0,
           "min_p": 0.0, "repetition_penalty": 1.0, "spec": 7, "spec_min_p": 0.72,
           "stop": [tk.ID_IM_END, tk.ID_ENDOFTEXT]}
    proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
    proc.stdin.flush()
    toks, done, prof = [], None, None
    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("EOF: " + "\n".join(err[-5:]))
        s = line.strip()
        if s.startswith("TOK "):
            toks.append(int(s.split()[-1]))
        elif s.startswith("DONE "):
            done = json.loads(s[5:])
            break
    wall = time.time() - t0
    try:
        proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=20)
    except Exception:
        proc.kill()
    for ln in err:
        if "DEC-PROF" in ln:
            prof = ln.strip()
    text = tk.decode(toks)
    out = {"tag": a.tag, "gen": done.get("generated"), "finish": done.get("finish"),
           "tps": done.get("decode_tps"), "tokwin": done.get("decode_tps") and None,
           "wall_s": round(wall, 1), "prof": prof, "toks": toks, "text": text,
           "vram_hit": done.get("vram_hit"), "promoted": done.get("promoted")}
    print("%s: gen=%s finish=%s tps=%.2f wall=%.1fs vram_hit=%.4f" % (
        a.tag, done.get("generated"), done.get("finish"), done.get("decode_tps"), wall,
        done.get("vram_hit")))
    print("  %s" % prof)
    if a.save:
        json.dump(out, open(a.save, "w"), indent=1)
        print("  saved %s" % a.save)


if __name__ == "__main__":
    main()
