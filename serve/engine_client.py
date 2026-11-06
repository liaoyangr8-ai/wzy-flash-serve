"""Multiplexed client for the concurrent batch engine (wzy_batch_engine).

The single-stream engine answers one request at a time and the supervisor talks to it by writing a
JSON line and reading its stdout until DONE. That model cannot overlap requests, so the batch engine
adds a request id to every line:

    {"req":7,"ids":[...],...}      ->  PREFILL {"req":7,...}
                                       TOK 7 1234
                                       DONE {"req":7,...}

One reader thread owns stdout and routes each line to the queue registered for its req id; HTTP
handler threads submit concurrently and each blocks on its own queue. Nothing here touches the
engine lock: the supervisor only needs it for lifecycle operations (start/stop).
"""

import json
import queue
import threading
import time
import uuid


class EngineClient:
    """Owns the engine's stdin/stdout once READY has been consumed by the boot worker."""

    def __init__(self, proc, log=None):
        self.proc = proc
        self._log = log or (lambda line: None)
        self._wlock = threading.Lock()
        self._lock = threading.Lock()
        self._queues = {}
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, name="engine-stdout", daemon=True)
        self._reader.start()

    def register(self, rid):
        q = queue.Queue()
        with self._lock:
            self._queues[rid] = q
        return q

    def unregister(self, rid):
        with self._lock:
            self._queues.pop(rid, None)

    def submit(self, rid, payload):
        """payload is the JSON request object (dict); the req id is injected here."""
        payload = dict(payload)
        payload["req"] = rid
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._wlock:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()

    def cancel(self, rid):
        with self._wlock:
            self.proc.stdin.write("CANCEL %d\n" % rid)
            self.proc.stdin.flush()

    def next_line(self, rid, timeout=600.0):
        with self._lock:
            q = self._queues.get(rid)
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return "__TIMEOUT__"

    @staticmethod
    def _rid_of(line):
        if line.startswith("TOK "):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return -1
            return -1
        if line.startswith("PREFILL ") or line.startswith("DONE "):
            try:
                return int(json.loads(line[line.index(" ") + 1:]).get("req", 0))
            except Exception:
                return -1
        return -1

    def _read_loop(self):
        try:
            for raw in self.proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                rid = self._rid_of(line)
                with self._lock:
                    q = self._queues.get(rid)
                if q is not None:
                    q.put(line)
                # PONG / stray lines have no queue; dropping them is correct.
        except Exception as exc:  # pragma: no cover - pipe teardown
            self._log("[wzy-serve] engine stdout reader stopped: %s" % exc)
        finally:
            self._closed = True
            with self._lock:
                for q in self._queues.values():
                    q.put(None)


def _rec_of(done_json, meta, req_id, generated, pf_data, wallclock_s, t_start):
    rec = dict(done_json)
    rec["id"] = req_id
    rec["path"] = meta.get("path", "/v1/chat/completions")
    rec["stream"] = bool(meta.get("stream", True))
    rec["state"] = "completed"
    rec["generated"] = max(int(rec.get("generated", 0)), generated)
    p_tok = int(rec.get("prompt_tokens", 0))
    reused = int(rec.get("reused", pf_data.get("reused", 0)))
    rec["read"] = pf_data.get("read", max(1, p_tok - reused))
    rec["prefill_tps"] = pf_data.get("prompt_tps", 0.0)
    rec["wallclock_s"] = round(wallclock_s, 2)
    rec["started_at"] = t_start
    rec["time"] = time.time()
    if rec.get("decode_ms", 0.0) > 0:
        rec["decode_tps"] = round(1000.0 * rec["generated"] / rec["decode_ms"], 2)
    return rec


def generate_stream_batch(sup, ids, params, meta=None, loop_guard=None):
    """Same event shape as EngineSupervisor.generate_stream, but transport is per-request.

    Yields ('prefill', dict) | ('token', int) | ('done', dict).
    """
    meta = meta or {}
    req_id = meta.get("id") or ("req-" + uuid.uuid4().hex[:12])
    rid = sup.next_rid()

    req_obj = {
        "ids": ids,
        "max_tokens": int(params.get("max_tokens", 512)),
        "temperature": float(params.get("temperature", 0.3)),
        "top_p": float(params.get("top_p", 0.75)),
        "top_k": int(params.get("top_k", 15)),
        "min_p": float(params.get("min_p", 0.02)),
        "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
        "stop": params.get("stop_ids", []),
    }
    if params.get("emb_file"):
        req_obj["emb_file"] = params["emb_file"]
        req_obj["emb_starts"] = params.get("emb_starts") or []
        req_obj["emb_counts"] = params.get("emb_counts") or []

    client = sup.client
    if client is None:
        raise RuntimeError("Engine client is not attached (engine not READY)")

    q = client.register(rid)
    t_start = time.time()
    pf_data = {}
    generated = 0
    done_received = False
    loop_tripped = False

    st = {
        "id": req_id,
        "path": meta.get("path", "/v1/chat/completions"),
        "phase": "prefill",
        "prompt_tokens": len(ids),
        "reused": 0,
        "read": len(ids),
        "prompt_ms": 0.0,
        "prompt_tps": 0.0,
        "generated": 0,
        "max_tokens": req_obj["max_tokens"],
        "inst_tps": 0.0,
        "avg_tps": 0.0,
        "elapsed_s": 0.0,
        "started": t_start,
        "decode_started": t_start,
    }
    sup.busy = True
    with sup.stats_lock:
        sup.busy_count += 1
        sup.req_stats[rid] = st
        sup.current_stat = st

    try:
        client.submit(rid, req_obj)
        while True:
            line = client.next_line(rid)
            if line is None:
                raise RuntimeError("Engine EOF during generation")
            if line == "__TIMEOUT__":
                client.cancel(rid)
                raise RuntimeError("Engine produced no output for %d s" % 600)

            if line.startswith("PREFILL "):
                pf_data = json.loads(line[8:])
                now_pf = time.time()
                st["phase"] = "answering"
                st["reused"] = pf_data.get("reused", 0)
                st["read"] = pf_data.get("read", max(1, len(ids) - pf_data.get("reused", 0)))
                st["prompt_ms"] = pf_data.get("prompt_ms", 0.0)
                st["prompt_tps"] = pf_data.get("prompt_tps", 0.0)
                st["decode_started"] = now_pf
                sup.last_prefill = dict(pf_data, time=now_pf)
                yield ("prefill", pf_data)

            elif line.startswith("TOK "):
                tid = int(line.split()[2])
                generated += 1
                st["generated"] = generated
                if loop_guard is not None and not loop_tripped and loop_guard.push(tid):
                    loop_tripped = True
                    st["loop_guard"] = True
                    sup._log("[wzy-serve] LOOP GUARD tripped on %s after %d tokens; cancelling %d"
                             % (req_id, generated, rid))
                    client.cancel(rid)
                yield ("token", tid)

            elif line.startswith("DONE "):
                done_received = True
                dn = json.loads(line[5:])
                wall = time.time() - t_start
                rec = _rec_of(dn, meta, req_id, generated, pf_data, wall, t_start)
                if loop_tripped:
                    rec["loop_guard"] = True
                    rec["finish"] = "stop"
                with sup.stats_lock:
                    sup.totals["requests"] += 1
                    sup.totals["prompt_tokens"] += int(rec.get("prompt_tokens", 0))
                    sup.totals["reused_tokens"] += int(rec.get("reused", 0))
                    sup.totals["read_tokens"] += int(rec.get("read", 0))
                    sup.totals["output_tokens"] += int(rec.get("generated", 0))
                    sup.totals["prefill_ms"] += float(rec.get("prompt_ms", 0.0))
                    sup.totals["decode_ms"] += float(rec.get("decode_ms", 0.0))
                sup.recent_requests.appendleft(rec)
                sup._store_detail(req_id, dict(rec, input=meta.get("input_preview", ""),
                                               reasoning="", output=""))
                yield ("done", rec)
                break
            else:
                continue
    finally:
        if not done_received and client is not None:
            try:
                client.cancel(rid)
            except Exception:
                pass
        client.unregister(rid)
        with sup.stats_lock:
            sup.busy_count = max(0, sup.busy_count - 1)
            sup.busy = sup.busy_count > 0
            sup.req_stats.pop(rid, None)
            if sup.current_stat is st:
                sup.current_stat = next(iter(sup.req_stats.values()), {})
