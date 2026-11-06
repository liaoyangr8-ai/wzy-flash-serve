# -*- coding: utf-8 -*-
"""wzy-flash-serve -- Production OpenAI-compatible HTTP/SSE server, Strata-style Monitor WebUI & Engine Launcher."""
import argparse
import collections
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import hashlib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB_DIR = os.path.join(HERE, "web")
sys.path.insert(0, HERE)
import tokenizer as tk  # noqa: E402
import frontend as fe  # noqa: E402
import responses as rp  # noqa: E402
import vision as vision_mod  # noqa: E402
import engine_client as engine_client_mod  # noqa: E402

class LoopGuard:
    """Detect degenerate repetition in a token stream.

    A healthy completion does not emit the same long gram three times inside a few hundred
    tokens. When it does, the model has fallen into a repetition attractor and every further
    token is wasted work.
    """

    def __init__(self, gram=48, repeats=3, window=768):
        self.gram = max(2, int(gram))
        self.repeats = max(2, int(repeats))
        self.window = max(self.gram * self.repeats, int(window))
        self.buf = []
        self.since = 0

    def push(self, tid):
        """Append one token id; return True when the stream is degenerate."""
        self.buf.append(int(tid))
        if len(self.buf) > self.window:
            del self.buf[: len(self.buf) - self.window]
        self.since += 1
        if len(self.buf) < self.gram * self.repeats or self.since < self.gram:
            return False
        # Only scan once per gram-sized stride: cheap and still catches the attractor fast.
        self.since = 0
        g = self.gram
        tail = self.buf[-g:]
        hits = 0
        n = len(self.buf)
        for i in range(n - g, -1, -1):
            if self.buf[i:i + g] == tail:
                hits += 1
                if hits >= self.repeats:
                    return True
        return False


def loop_guard_config():
    """(enabled, gram, repeats, window) from os.environ, then config.json, then defaults."""
    sec = CONFIG.get("loop_guard") if isinstance(CONFIG, dict) else None
    sec = sec if isinstance(sec, dict) else {}

    def num(key, env, default):
        if env in os.environ:
            try:
                return int(os.environ[env])
            except ValueError:
                pass
        if key in sec:
            try:
                return int(sec[key])
            except (TypeError, ValueError):
                pass
        return default

    enabled = True
    if "WZY_LOOP_GUARD" in os.environ:
        enabled = os.environ["WZY_LOOP_GUARD"].strip() not in ("0", "false", "False", "off")
    elif "enabled" in sec:
        enabled = bool(sec["enabled"])
    return (enabled,
            num("gram", "WZY_LOOP_GRAM", 48),
            num("repeats", "WZY_LOOP_REPEATS", 3),
            num("window", "WZY_LOOP_WINDOW", 768))


def read_host_sys():
    """Read Linux /proc/meminfo & /proc/stat with zero external dependencies."""
    ram_total_gib = 94.3
    ram_used_gib = 0.0
    ram_pct = 0.0
    try:
        mem = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = int(parts[1].strip().split()[0])  # kB
                    mem[k] = v
        tot_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
        if tot_kb > 0:
            ram_total_gib = round(tot_kb / 1048576.0, 1)
            ram_used_gib = round(max(0, tot_kb - avail_kb) / 1048576.0, 1)
            ram_pct = round(100.0 * (tot_kb - avail_kb) / tot_kb, 1)
    except Exception:
        pass
    return {
        "ram_used_gib": ram_used_gib,
        "ram_total_gib": ram_total_gib,
        "ram_pct": ram_pct,
        "cores": os.cpu_count() or 32,
    }


_GPU_CACHE = {"ts": 0.0, "data": []}
_GPU_LOCK = threading.Lock()


def query_gpus(max_age=0.8):
    now = time.time()
    with _GPU_LOCK:
        if now - _GPU_CACHE["ts"] < max_age and _GPU_CACHE["data"]:
            return _GPU_CACHE["data"]
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
        )
        res = []
        for ln in out.strip().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 8:
                sm_mhz = int(float(p[8])) if len(p) >= 9 and p[8].replace(".", "").isdigit() else 0
                res.append(
                    {
                        "index": int(p[0]),
                        "name": p[1],
                        "mem_mib": int(float(p[2])),
                        "mem_total_mib": int(float(p[3])),
                        "util": int(float(p[4])),
                        "temp": int(float(p[5])),
                        "power_w": round(float(p[6]), 1),
                        "power_limit_w": round(float(p[7]), 1),
                        "sm_mhz": sm_mhz,
                    }
                )
        with _GPU_LOCK:
            _GPU_CACHE["ts"] = now
            _GPU_CACHE["data"] = res
        return res
    except Exception:
        return _GPU_CACHE["data"]


class EngineSupervisor:
    """Supervises the dual-GPU C++/CUDA PP2 process with full Start/Stop/Restart Launcher & 1Hz Telemetry."""

    def __init__(self, cfg, config_path):
        self.cfg = cfg
        self.config_path = config_path
        self.bin_path = cfg["engine_bin"]
        # Concurrent batch engine (optional): a separate binary that keeps N sessions resident and
        # decodes them on private streams. When enabled the supervisor stops serialising requests and
        # talks to the engine through a multiplexed client keyed by request id.
        batch_cfg = cfg.get("batch") or {}
        self.batch_mode = bool(batch_cfg.get("enabled", False))
        self.batch_slots = int(batch_cfg.get("slots", 2))
        self.batch_ctx = int(batch_cfg.get("ctx", 32768))
        if self.batch_mode and batch_cfg.get("engine_bin"):
            self.bin_path = batch_cfg["engine_bin"]
        self.client = None
        self.stats_lock = threading.Lock()
        self.busy_count = 0
        # Batch mode runs several requests at once; each keeps its own telemetry record and
        # `current_stat` points at the most recently admitted one.
        self.req_stats = {}
        self._rid_lock = threading.Lock()
        self._rid = 0
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.proc = None
        self.state = "stopped"  # "stopped" | "booting" | "ready" | "error"
        self.boot_info = {"stage": "未启动", "pct": 0, "started": 0.0, "elapsed_s": 0.0, "error": ""}
        self.ready_info = {}
        self.busy = False
        self.current_stat = {}
        self.last_prefill = {}
        self.recent_requests = collections.deque(maxlen=100)
        self.request_details = collections.OrderedDict()
        self.engine_logs = collections.deque(maxlen=300)
        self.tps_series = collections.deque(maxlen=120)
        self._tok_times = collections.deque(maxlen=256)
        self._ready_ev = threading.Event()
        self._manual_stop = False
        self.totals = {
            "since": time.time(),
            "requests": 0,
            "prompt_tokens": 0,
            "reused_tokens": 0,
            "read_tokens": 0,
            "output_tokens": 0,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "park_restores": 0,
        }
        self.session_slots = {"active_tokens": 0, "parked_tokens": 0, "park_restores": 0}
        self.conv_cache = collections.OrderedDict()
        threading.Thread(target=self._telemetry_loop, daemon=True).start()
        # Pictures: Strata's CPU reader (`strata-vision` + the model's mmproj) -> embeddings that the
        # engine drops onto the prompt's `<|image_pad|>` tokens.  Off when config.json has no "vision".
        self.vision = vision_mod.Vision(cfg.get("vision"))

        # Boot engine asynchronously on startup so WebUI is reachable at 0.0s!
        if cfg.get("start_on_boot", True):
            self.start_async()

    def _log(self, line):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {line}"
        self.engine_logs.append(entry)
        print(line, flush=True)

    def _build_env(self):
        env = os.environ.copy()
        for k, v in self.cfg.get("env", {}).items():
            env[k] = str(v)
        if self.batch_mode:
            # The batch engine has no 256K host KV mirror: its ring is the whole context. Sessions
            # share one expert cache, so WZY_LEAVE_MIB is set by the engine itself from WZY_BATCH.
            env["WZY_CTX"] = str(self.batch_ctx)
            env["WZY_BATCH"] = str(self.batch_slots)
            env.setdefault("WZY_CPU_THREADS", "14")
        return env

    def next_rid(self):
        with self._rid_lock:
            self._rid += 1
            if self._rid >= 2000000000:
                self._rid = 1
            return self._rid

    def _loop_guard(self):
        on, gram, repeats, window = loop_guard_config()
        return LoopGuard(gram, repeats, window) if on else None

    def save_config(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log(f"[wzy-serve] warning: save_config failed: {e}")

    def start_async(self):
        with self.state_lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": True, "state": self.state, "message": "引擎已在运行中"}
            self._manual_stop = False
            self.state = "booting"
            self._ready_ev.clear()
            self.boot_info = {
                "stage": "正在拉起双卡 CUDA PP2 引擎进程...",
                "pct": 8,
                "started": time.time(),
                "elapsed_s": 0.0,
                "error": "",
            }
        threading.Thread(target=self._boot_worker, daemon=True).start()
        return {"ok": True, "state": "booting", "message": "正在后台启动双卡引擎"}

    def _stderr_reader(self, proc):
        try:
            for raw in iter(proc.stderr.readline, ""):
                if not raw:
                    break
                s = raw.strip()
                if not s:
                    continue
                self._log(s)
                if "Loaded MTP drafter" in s:
                    self.boot_info["stage"] = "已加载 GPU1 MTP Q2_0 投机草稿头与 114K 中文词表"
                    self.boot_info["pct"] = max(self.boot_info["pct"], 22)
                elif "VRAM expert slots" in s:
                    self.boot_info["stage"] = "已分配双卡 GDDR6 专家槽位与 32K+4K Sink KV 环"
                    self.boot_info["pct"] = max(self.boot_info["pct"], 38)
                elif "Pre-faulting & pinning" in s:
                    self.boot_info["stage"] = "正在 16 线程预缺页并锁页映射 63.3 GiB 专家包 (约 45s)..."
                    self.boot_info["pct"] = max(self.boot_info["pct"], 55)
                elif "GPU0 prefilled" in s:
                    self.boot_info["stage"] = "GPU0 显存热门专家预填充完成，正在填充 GPU1..."
                    self.boot_info["pct"] = max(self.boot_info["pct"], 82)
                elif "GPU1 prefilled" in s:
                    self.boot_info["stage"] = "双卡显存预填充完成，正在捕获 CUDA Graph..."
                    self.boot_info["pct"] = max(self.boot_info["pct"], 94)
        except Exception:
            pass

    def _boot_worker(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            env = self._build_env()
            self._log(
                f"[wzy-serve] spawning engine: {self.bin_path} "
                f"(split={env.get('WZY_SPLIT', '24')}, ctx={env.get('WZY_CTX', '262144')}, adapt={env.get('WZY_ADAPT_SLOTS', '2048')})"
            )
            try:
                proc = subprocess.Popen(
                    [self.bin_path],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    preexec_fn=lambda: os.sched_setaffinity(0, set(range(os.cpu_count() or 16)))
                    if hasattr(os, "sched_setaffinity")
                    else None,
                )
            except Exception as e:
                self.state = "error"
                self.boot_info["error"] = str(e)
                self.boot_info["stage"] = f"启动失败: {e}"
                self._ready_ev.set()
                return

            self.proc = proc
            if hasattr(os, "sched_setaffinity"):
                try:
                    os.sched_setaffinity(0, {13, 14})
                except OSError:
                    pass

            threading.Thread(target=self._stderr_reader, args=(proc,), daemon=True).start()

            while True:
                line = proc.stdout.readline()
                if not line:
                    rc = proc.poll()
                    if self._manual_stop:
                        self.state = "stopped"
                        self.boot_info["stage"] = "已停止 (0 MiB 显存占用)"
                        self.boot_info["pct"] = 0
                    else:
                        self.state = "error"
                        self.boot_info["error"] = f"引擎启动期间退出 (code={rc})"
                        self.boot_info["stage"] = self.boot_info["error"]
                        self._log(f"[wzy-serve] ERROR: {self.boot_info['error']}")
                    self._ready_ev.set()
                    return
                s = line.strip()
                if s.startswith("READY "):
                    self.ready_info = json.loads(s[6:])
                    self.state = "ready"
                    self.boot_info["stage"] = "就绪 (Ready)"
                    self.boot_info["pct"] = 100
                    self.boot_info["elapsed_s"] = round(time.time() - self.boot_info["started"], 2)
                    self.conv_cache.clear()
                    self.session_slots = {"active_tokens": 0, "parked_tokens": 0, "park_restores": 0}
                    if self.batch_mode:
                        # Hand stdout over to the multiplexed reader: from here on, request handlers
                        # submit concurrently and wait on their own queues.
                        self.client = engine_client_mod.EngineClient(proc, self._log)
                    self._log(f"[wzy-serve] engine READY: {self.ready_info}")
                    self._ready_ev.set()
                    return

    def stop_engine(self, force=False, wait_s=0.0):
        wait_t = max(0.0, min(300.0, float(wait_s or 0.0)))
        acquired = self.lock.acquire(timeout=wait_t) if wait_t > 0.0 else self.lock.acquire(blocking=False)
        if not acquired:
            if not force:
                return {
                    "ok": False,
                    "status": "busy",
                    "busy": True,
                    "state": self.state,
                    "message": "引擎正在处理推理请求，已拒绝卸载以保护活动请求（可传 wait 等待或 force=true 强制停止）",
                }
        try:
            if self.busy and not force:
                return {
                    "ok": False,
                    "status": "busy",
                    "busy": True,
                    "state": self.state,
                    "message": "引擎正在处理推理请求，已拒绝卸载以保护活动请求",
                }
            self._manual_stop = True
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.proc = None
                self.client = None
                self.state = "stopped"
                self.ready_info = {}
                self.busy = False
                self.current_stat = {}
                self.req_stats.clear()
                self.boot_info["stage"] = "已停止 (显存已释放)"
                self.boot_info["pct"] = 0
                return {"ok": True, "status": "unloaded", "state": "stopped", "message": "引擎已处于停止状态"}

            self._log(f"[wzy-serve] Stopping engine via Launcher request (force={force})...")
            try:
                if proc.stdin:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except Exception:
                    pass
            self.proc = None
            self.client = None
            self.state = "stopped"
            self.ready_info = {}
            self.busy = False
            self.busy_count = 0
            self.current_stat = {}
            self.req_stats.clear()
            self.conv_cache.clear()
            self.session_slots = {"active_tokens": 0, "parked_tokens": 0, "park_restores": 0}
            self.boot_info["stage"] = "已停止 (44 GiB 双卡显存已全部释放)"
            self.boot_info["pct"] = 0
            self._ready_ev.set()
            self._log("[wzy-serve] Engine stopped cleanly; GPU VRAM released.")
            return {"ok": True, "status": "unloaded", "state": "stopped", "message": "双卡引擎已关闭，显存已释放"}
        finally:
            if acquired:
                self.lock.release()

    def restart_engine(self, force=False, wait_s=0.0):
        stop_res = self.stop_engine(force=force, wait_s=wait_s)
        if not stop_res.get("ok", False):
            return stop_res
        time.sleep(0.4)
        return self.start_async()
    def ensure_alive(self):
        if self.state == "booting":
            self._ready_ev.wait(timeout=180.0)
        if self.proc is not None and self.proc.poll() is None and self.state == "ready":
            return True
        if self._manual_stop and not self.cfg.get("auto_load", False):
            raise RuntimeError("Engine is currently stopped via WebUI Launcher. Click '启动引擎' in WebUI or enable Auto-Load.")
        self._log("[wzy-serve] Auto-starting engine for incoming request...")
        self.start_async()
        self._ready_ev.wait(timeout=180.0)
        if self.state != "ready" or self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"Engine failed to start: {self.boot_info.get('error') or self.boot_info.get('stage')}")
        return True

    def save_profile(self):
        """Persist the learned expert heat into expert-profile-cjk.bin (engine `SAVE_PROFILE` command)."""
        if self.batch_mode:
            # The batch engine has no SAVE_PROFILE command; answering here would steal a line from the
            # multiplexed stdout reader. Report "not supported" instead of hanging.
            return False
        with self.lock:
            if self.proc is None or self.proc.poll() is not None or self.state != "ready":
                return False
            try:
                self.proc.stdin.write("SAVE_PROFILE\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                return line.strip() == "OK"
            except Exception as e:
                self._log(f"[wzy-serve] save_profile failed: {e}")
                return False
    def reset_state(self):
        if self.batch_mode:
            # Sessions stay resident; there is nothing to reset on the engine side.
            with self.lock:
                self.conv_cache.clear()
                self.session_slots = {"active_tokens": 0, "parked_tokens": 0, "park_restores": 0}
            return True
        with self.lock:
            self.conv_cache.clear()
            self.session_slots = {"active_tokens": 0, "parked_tokens": 0, "park_restores": 0}
            if self.proc is None or self.proc.poll() is not None or self.state != "ready":
                return True
            self.proc.stdin.write("RESET\n")
            self.proc.stdin.flush()
            return self.proc.stdout.readline().strip() == "OK"

    _DSH_TURN_RE = re.compile(
        r"Time sampled while preparing turn \d+, step \d+: (\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}(?:\.\d+)?([^\n]*)"
    )
    _DSH_ELAPSED_RE = re.compile(r"Elapsed since the preceding model-visible message: [^\n]+")

    @classmethod
    def _norm_text(cls, txt):
        if not txt or "Time sampled while preparing turn " not in txt:
            return txt
        txt = cls._DSH_TURN_RE.sub(r"Time sampled while preparing turn 1, step 1: \1T00:00:00\2", txt)
        txt = cls._DSH_ELAPSED_RE.sub("Elapsed since the preceding model-visible message: unavailable.", txt)
        return txt

    @classmethod
    def normalize_messages(cls, msgs):
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                out.append(m)
                continue
            mc = dict(m)
            content = m.get("content")
            if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image" for p in content):
                # Keep the template's list form: its image items must reach the template, which writes the
                # `<|vision_start|><|image_pad|><|vision_end|>` markers for them.
                mc["content"] = [dict(p, text=cls._norm_text(p.get("text"))) if isinstance(p, dict) and p.get("type") == "text"
                                 else p for p in content]
            else:
                mc["content"] = cls._norm_text(tk._render_content(content))
            out.append(mc)
        return out
    @classmethod
    def _msg_sig(cls, mlist):
        sig = []
        for m in mlist:
            r = m.get("role", "")
            c = cls._norm_text(tk._render_content(m.get("content"))).strip()
            norm_tc = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                if isinstance(fn, dict):
                    norm_tc.append({"name": fn.get("name", ""), "arguments": fe.tool_arguments(fn.get("arguments"))})
            tc_s = json.dumps(norm_tc, sort_keys=True, ensure_ascii=False) if norm_tc else ""
            sig.append((r, c, tc_s))
        return tuple(sig)

    def build_chat_ids(self, msgs, tools, kwargs, force=None, allow_cache=True):
        norm_msgs = self.normalize_messages(msgs)
        tools_sig = json.dumps(tools or [], sort_keys=True, ensure_ascii=False)
        if allow_cache and len(norm_msgs) >= 3 and self.conv_cache:
            for cut in range(len(norm_msgs) - 1, 0, -1):
                if any(m.get("role") == "system" for m in norm_msgs[cut:]):
                    continue
                prefix_sig = self._msg_sig(norm_msgs[:cut])
                cached_ids = self.conv_cache.get((tools_sig, prefix_sig))
                if cached_ids is not None:
                    self.conv_cache.move_to_end((tools_sig, prefix_sig))
                    dummy = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
                    base_ids = tk.render_and_encode(dummy, tools=None, add_generation_prompt=False, **kwargs)
                    full_ids = tk.render_and_encode(dummy + norm_msgs[cut:], tools=None, add_generation_prompt=True, **kwargs)
                    tail_ids = [tk.ID_IM_END] + tk.encode("\n", allow_special=False) + full_ids[len(base_ids) :]
                    if force:
                        tail_ids += tk.encode(force, allow_special=True)
                    return list(cached_ids) + tail_ids
        ids = tk.render_and_encode(norm_msgs, tools=tools, add_generation_prompt=True, **kwargs)
        if force:
            ids += tk.encode(force, allow_special=True)
        return ids

    def record_conv_ids(self, msgs, tools, ans_text, tool_calls, full_ids):
        try:
            norm_msgs = self.normalize_messages(msgs)
            norm_calls = []
            for c in tool_calls or []:
                if hasattr(c, "name"):
                    norm_calls.append({"function": {"name": c.name, "arguments": c.arguments}})
                elif isinstance(c, dict):
                    fn = c.get("function", c)
                    norm_calls.append({"function": {"name": fn.get("name"), "arguments": fe.tool_arguments(fn.get("arguments"))}})
            asst_msg = {"role": "assistant", "content": (ans_text or "").strip()}
            if norm_calls:
                asst_msg["tool_calls"] = norm_calls
            full_sig = self._msg_sig(norm_msgs + [asst_msg])
            tools_sig = json.dumps(tools or [], sort_keys=True, ensure_ascii=False)
            key = (tools_sig, full_sig)
            self.conv_cache[key] = list(full_ids)
            self.conv_cache.move_to_end(key)
            while len(self.conv_cache) > 16:
                self.conv_cache.popitem(last=False)
        except Exception:
            pass

    # ---- vision (pictures) --------------------------------------------------------------
    # The chat template already renders one `<|vision_start|><|image_pad|><|vision_end|>` for every image item of
    # a message (`fe._parts_of` keeps the item, its source normalised to a data URI / URL / path), so the server
    # only has to read the pictures and expand each single pad into its run of pads.
    def prepare_vision(self, msgs):
        """-> (messages unchanged, the image sources in prompt order)."""
        sources = fe.images_of(msgs) if isinstance(msgs, list) else []
        if sources and not self.vision.enabled:
            raise ValueError('this server has no vision encoder configured (config.json "vision")')
        return msgs, sources
    def vision_ids(self, ids, sources):
        """Read the pictures (CPU mmproj reader), expand each single pad into its run. -> (ids, emb)."""
        if not sources:
            return ids, {}
        if not self.vision.enabled:
            raise ValueError('this server has no vision encoder configured (config.json "vision")')
        encoded = []
        for src in sources:
            encoded.append(self.vision.encode(vision_mod.Vision.decode_source(src)))
        counts = [n for _, n in encoded]
        pad = tk.SPECIAL_TOKENS[vision_mod.IMAGE_PAD]
        vstart = tk.SPECIAL_TOKENS[vision_mod.VISION_START]
        # Its `count` pads (and the rows that replace their embeddings) start after the `<|vision_start|>`.
        out, runs, k = [], [], 0
        for j, t in enumerate(ids):
            if t == pad and j > 0 and ids[j - 1] == vstart:
                if k >= len(counts):
                    raise ValueError("the prompt and its images do not match")
                runs.append(len(out))
                out.extend([pad] * counts[k])
                k += 1
            else:
                out.append(t)
        if k != len(counts):
            raise ValueError("the prompt and its images do not match")
        key = hashlib.sha256(repr((runs, counts)).encode() + "".join(str(p) for p, _ in encoded).encode()).hexdigest()[:24]
        path, _total = self.vision.combined(encoded, key)
        self._log(f"[wzy-vision] {len(counts)} image(s) -> {sum(counts)} tokens at {runs}")
        return out, {"emb_file": str(path), "emb_starts": runs, "emb_counts": counts}

    def run_parsed_events(self, ids, params, thinking, tools, force=None, meta=None):
        """Runs engine token stream through `fe.OutputParser` (streaming tool_start/tool_args/tool_call, <think>, bare recovery)."""
        parser = fe.OutputParser(
            thinking=thinking,
            tools=tools,
            stream_tools=True,
            recover=True,
            repair_names=True,
        )
        detok = tk.IncrementalDecoder(skip_special=True, keep_tool_tags=True)
        raw_gen_ids = []
        if force and not thinking:
            for ev in parser.feed(force):
                yield ("event", ev)
        pf_info = {}
        for kind, data in self.generate_stream(ids, params, meta=meta):
            if kind == "prefill":
                pf_info = data
                yield ("prefill", pf_info)
            elif kind == "token":
                tid = data
                raw_gen_ids.append(tid)
                if tid == tk.ID_THINK_START and parser.state == "reasoning" and len(raw_gen_ids) <= 2:
                    continue
                piece = detok.push(tid)
                if piece:
                    for ev in parser.feed(piece):
                        yield ("event", ev)
            elif kind == "done":
                tail = detok.flush()
                if tail:
                    for ev in parser.feed(tail):
                        yield ("event", ev)
                fin = "stop" if data.get("finish") == "cancelled" else data.get("finish", "stop")
                for ev in parser.finish(fin):
                    yield ("event", ev)
                yield ("done", (data, pf_info, raw_gen_ids))

    def _store_detail(self, req_id, rec):
        self.request_details[req_id] = rec
        while len(self.request_details) > 100:
            self.request_details.popitem(last=False)

    def _telemetry_loop(self):
        while True:
            time.sleep(1.0)
            now = time.time()
            try:
                if self.state == "booting" and self.boot_info.get("started"):
                    self.boot_info["elapsed_s"] = round(now - self.boot_info["started"], 1)
                    # Smoothly advance progress between 55% and 90% during 45s host pin + VRAM prefill
                    if 55 <= self.boot_info["pct"] < 90:
                        self.boot_info["pct"] = min(90, 55 + int(self.boot_info["elapsed_s"] * 0.6))

                inst_tps = 0.0
                pf_tps = 0.0
                phase = self.state
                if self.busy and self.current_stat:
                    phase = self.current_stat.get("phase", "generating")
                    self.current_stat["elapsed_s"] = round(now - self.current_stat.get("started", now), 1)
                    if phase == "answering":
                        # Compute 2.0-second sliding-window decode tok/s
                        cutoff = now - 2.0
                        while self._tok_times and self._tok_times[0] < cutoff:
                            self._tok_times.popleft()
                        if len(self._tok_times) >= 2:
                            span = max(0.25, now - self._tok_times[0])
                            inst_tps = round(len(self._tok_times) / span, 2)
                        dec_t0 = self.current_stat.get("decode_started", self.current_stat.get("started", now))
                        dec_el = max(0.05, now - dec_t0)
                        avg_tps = round(self.current_stat.get("generated", 0) / dec_el, 2)
                        if inst_tps <= 0.0:
                            inst_tps = avg_tps
                        self.current_stat["inst_tps"] = inst_tps
                        self.current_stat["avg_tps"] = avg_tps
                    elif phase == "prefill":
                        pf_tps = self.last_prefill.get("prompt_tps", 0.0)

                gpus = query_gpus(max_age=0.9)
                g0_u = gpus[0]["util"] if len(gpus) > 0 else 0
                g1_u = gpus[1]["util"] if len(gpus) > 1 else 0
                self.tps_series.append(
                    {
                        "t": round(now, 1),
                        "decode_tps": inst_tps,
                        "prefill_tps": round(self.current_stat.get("prompt_tps", 0.0), 1) if self.busy else 0.0,
                        "gpu0_util": g0_u,
                        "gpu1_util": g1_u,
                        "phase": phase,
                    }
                )
            except Exception:
                pass

    def generate_stream(self, ids, params, meta=None):
        """Yields ('prefill', dict) | ('token', int) | ('done', dict) under engine lock."""
        if self.batch_mode:
            # Concurrent path: requests are multiplexed by id, so handlers must NOT serialise on
            # self.lock here. Lifecycle (start/stop) still takes it.
            yield from engine_client_mod.generate_stream_batch(
                self, ids, params, meta, self._loop_guard())
            return
        meta = meta or {}
        req_id = meta.get("id") or ("req-" + uuid.uuid4().hex[:12])
        req_obj = {
            "ids": ids,
            "max_tokens": int(params.get("max_tokens", 512)),
            "temperature": float(params.get("temperature", 0.3)),
            "top_p": float(params.get("top_p", 0.75)),
            "top_k": int(params.get("top_k", 15)),
            "min_p": float(params.get("min_p", 0.02)),
            "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
            "spec": int(params.get("spec", 7)),
            "spec_min_p": float(params.get("spec_min_p", 0.72)),
            "no_reuse": 1 if params.get("no_reuse") else 0,
            "stop": params.get("stop_ids", [tk.ID_IM_END, tk.ID_ENDOFTEXT]),
        }
        # Pictures: the reader's rows, and where each image's `<|image_pad|>` run starts in the prompt.
        if params.get("emb_file"):
            req_obj["emb_file"] = params["emb_file"]
            req_obj["emb_starts"] = params.get("emb_starts") or []
            req_obj["emb_counts"] = params.get("emb_counts") or []
        payload = json.dumps(req_obj, separators=(",", ":")) + "\n"
        with self.lock:
            self.ensure_alive()
            self.busy = True
            t_start = time.time()
            self._tok_times.clear()
            self.current_stat = {
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
            pf_data = {}
            done_received = False
            loop_tripped = False
            _lg_on, _lg_gram, _lg_rep, _lg_win = loop_guard_config()
            loop_guard = LoopGuard(_lg_gram, _lg_rep, _lg_win) if _lg_on else None
            try:
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise RuntimeError("Engine EOF during generation")
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("PREFILL "):
                        pf_data = json.loads(s[8:])
                        now_pf = time.time()
                        self.current_stat["phase"] = "answering"
                        self.current_stat["reused"] = pf_data.get("reused", 0)
                        self.current_stat["read"] = pf_data.get("read", max(1, len(ids) - pf_data.get("reused", 0)))
                        self.current_stat["prompt_ms"] = pf_data.get("prompt_ms", 0.0)
                        self.current_stat["prompt_tps"] = pf_data.get("prompt_tps", 0.0)
                        self.current_stat["decode_started"] = now_pf
                        self.last_prefill = dict(pf_data, time=now_pf)
                        self.session_slots["parked_tokens"] = pf_data.get("parked_tokens", self.session_slots.get("parked_tokens", 0))
                        if self.tps_series:
                            self.tps_series[-1]["prefill_tps"] = round(pf_data.get("prompt_tps", 0.0), 1)
                        yield ("prefill", pf_data)
                    elif s.startswith("TOK "):
                        tid = int(s[4:])
                        now_t = time.time()
                        self._tok_times.append(now_t)
                        self.current_stat["generated"] += 1
                        if loop_guard is not None and not loop_tripped and loop_guard.push(tid):
                            loop_tripped = True
                            self.current_stat["loop_guard"] = True
                            self._log(
                                "[wzy-serve] LOOP GUARD tripped on %s after %d tokens "
                                "(a %d-token block repeated %d times within %d); aborting turn"
                                % (req_id, self.current_stat.get("generated", 0),
                                   _lg_gram, _lg_rep, _lg_win)
                            )
                            try:
                                if self.proc is not None and self.proc.poll() is None:
                                    os.kill(self.proc.pid, signal.SIGUSR1)
                            except Exception:
                                pass
                        yield ("token", tid)
                    elif s.startswith("DONE "):
                        done_received = True
                        now_done = time.time()
                        dn = json.loads(s[5:])
                        gen_actual = max(dn.get("generated", 0), self.current_stat.get("generated", 0))
                        dn["generated"] = gen_actual
                        if dn.get("decode_ms", 0.0) > 0:
                            dn["decode_tps"] = round(1000.0 * gen_actual / dn["decode_ms"], 2)
                        p_tok = dn.get("prompt_tokens", len(ids))
                        reused_tok = dn.get("reused", pf_data.get("reused", 0))
                        read_tok = pf_data.get("read", max(1, p_tok - reused_tok))
                        p_ms = dn.get("prompt_ms", pf_data.get("prompt_ms", 0.0))
                        pf_tps = pf_data.get("prompt_tps", round(1000.0 * read_tok / p_ms, 1) if p_ms > 0 else 0.0)

                        self.totals["requests"] += 1
                        self.totals["prompt_tokens"] += p_tok
                        self.totals["reused_tokens"] += reused_tok
                        self.totals["read_tokens"] += read_tok
                        self.totals["output_tokens"] += gen_actual
                        self.totals["prefill_ms"] += p_ms
                        self.totals["decode_ms"] += dn.get("decode_ms", 0.0)
                        if dn.get("restored_park"):
                            self.totals["park_restores"] = self.totals.get("park_restores", 0) + 1
                        self.session_slots = {
                            "active_tokens": dn.get("active_tokens", p_tok + dn.get("generated", 0)),
                            "parked_tokens": dn.get("parked_tokens", 0),
                            "park_restores": self.totals.get("park_restores", 0),
                            "conv_entries": len(self.conv_cache),
                        }
                        rec = dict(dn)
                        if loop_tripped:
                            # The engine reports "cancelled"; for the client this was a deliberate,
                            # successful stop, so surface it as a normal finish with a flag.
                            rec["loop_guard"] = True
                            rec["finish"] = "stop"
                        rec["id"] = req_id
                        rec["path"] = meta.get("path", "/v1/chat/completions")
                        rec["stream"] = bool(meta.get("stream", True))
                        rec["state"] = "completed"
                        rec["read"] = read_tok
                        rec["prefill_tps"] = round(pf_tps, 1)
                        rec["wallclock_s"] = round(now_done - t_start, 2)
                        rec["started_at"] = t_start
                        rec["time"] = now_done
                        self.recent_requests.appendleft(rec)
                        self._store_detail(
                            req_id,
                            dict(
                                rec,
                                input=meta.get("input_preview", ""),
                                reasoning="",
                                output="",
                            ),
                        )
                        yield ("done", rec)
                        break
            finally:
                if not done_received and self.proc is not None and self.proc.poll() is None:
                    self._log(f"[wzy-serve] Request {req_id} aborted by client; sending SIGUSR1 and draining stdout to keep pipe synchronized...")
                    try:
                        os.kill(self.proc.pid, signal.SIGUSR1)
                    except Exception:
                        pass
                    try:
                        while True:
                            rem = self.proc.stdout.readline()
                            if not rem or rem.strip().startswith("DONE "):
                                break
                    except Exception:
                        pass
                self.busy = False
                self.current_stat = {}
    def attach_request_text(self, req_id, reasoning_text, output_text, response_json=None):
        det = self.request_details.get(req_id)
        if det is not None:
            det["reasoning"] = (reasoning_text or "")[:16384]
            det["output"] = (output_text or "")[:32768]
            if response_json is not None:
                try:
                    det["response"] = json.dumps(response_json, ensure_ascii=False, indent=2)[:32768]
                except Exception:
                    pass

    def shutdown(self):
        self.stop_engine(force=True)

class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class PortManager:
    """Manages HTTP listeners dynamically so the user can change serving ports from the WebUI without restarting."""

    def __init__(self, host, handler_cls):
        self.host = host
        self.handler_cls = handler_cls
        self.lock = threading.Lock()
        self.servers = {}  # port -> (srv, thread)
        self.closing = {}  # port -> (srv, thread, timer)

    def active_ports(self):
        with self.lock:
            return sorted(self.servers.keys())

    def sync_ports(self, primary_port, mirror_ports=None, launcher_port=11439, defer_close_s=1.5):
        mirror_ports = mirror_ports or []
        want = []
        for p in [primary_port] + list(mirror_ports) + ([launcher_port] if launcher_port else []):
            p = int(p)
            if 1024 <= p <= 65535 and p not in want:
                want.append(p)

        started = []
        errors = []
        with self.lock:
            # 1. Bind any new ports first before closing old ones
            for p in want:
                if p in self.closing:
                    srv_c, th_c, tmr_c = self.closing.pop(p)
                    tmr_c.cancel()
                    self.servers[p] = (srv_c, th_c)
                if p not in self.servers:
                    try:
                        srv = ReusableThreadingHTTPServer((self.host, p), self.handler_cls)
                        th = threading.Thread(target=srv.serve_forever, daemon=True)
                        th.start()
                        self.servers[p] = (srv, th)
                        started.append(p)
                        print(f"[wzy-serve] HTTP listening on http://{self.host}:{p}", flush=True)
                    except OSError as e:
                        errors.append(f"端口 {p} 绑定失败: {e}")

            if errors and primary_port not in self.servers:
                return {"ok": False, "error": "; ".join(errors), "active_ports": sorted(self.servers.keys())}

            # 2. Schedule graceful close of removed ports after response completes
            to_remove = [p for p in list(self.servers.keys()) if p not in want]
            for p in to_remove:
                srv, th = self.servers.pop(p)

                def _close_srv(s=srv, port=p):
                    with self.lock:
                        self.closing.pop(port, None)
                    try:
                        s.shutdown()
                        s.server_close()
                        print(f"[wzy-serve] Closed previous HTTP listener on :{port}", flush=True)
                    except Exception:
                        pass

                tmr = threading.Timer(defer_close_s, _close_srv)
                self.closing[p] = (srv, th, tmr)
                tmr.start()
            return {
                "ok": len(errors) == 0,
                "port": int(primary_port),
                "launcher_port": int(launcher_port),
                "active_ports": sorted(self.servers.keys()),
                "started": started,
                "warnings": errors,
            }

    def shutdown_all(self):
        with self.lock:
            items = list(self.servers.items())
            self.servers.clear()
        for _, (srv, _) in items:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass


SUPERVISOR = None
PORT_MGR = None
CONFIG = {}


class LjqHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _cors(self):
        origin = (self.headers.get("Origin") or "").rstrip("/")
        allowed = CONFIG.get("cors_origins")
        if isinstance(allowed, list) and allowed and "*" not in allowed:
            host_hdr = (self.headers.get("Host") or "").rstrip("/")
            origin_host = origin.split("://", 1)[-1] if origin else ""
            if origin and (
                origin in allowed
                or origin_host == host_hdr
                or origin_host.startswith(("127.0.0.1:", "localhost:", "192.168."))
                or origin_host in ("127.0.0.1", "localhost")
            ):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
        else:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, OpenAI-Beta",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    def _read_body(self):
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            chunks = []
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                sz = int(line.split(b";", 1)[0].strip(), 16)
                if sz == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(sz))
                self.rfile.read(2)
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length > 0 else b"{}"
    def _send_json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, fpath, content_type=None):
        if not os.path.isfile(fpath):
            self._send_json({"error": {"message": "File not found"}}, 404)
            return
        with open(fpath, "rb") as f:
            raw = f.read()
        if not content_type:
            content_type, _ = mimetypes.guess_type(fpath)
            content_type = content_type or "application/octet-stream"
            if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
                content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def _check_auth(self):
        req_key = CONFIG.get("api_key")
        if not req_key or not CONFIG.get("enforce_api_key", False):
            return True
        auth = self.headers.get("Authorization", "")
        xkey = self.headers.get("x-api-key", "")
        return auth == f"Bearer {req_key}" or xkey == req_key

    def _check_admin_origin(self):
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if not origin:
            return True
        host_hdr = (self.headers.get("Host") or "").rstrip("/")
        origin_host = origin.split("://", 1)[-1]
        allowed = CONFIG.get("cors_origins") or []
        if (
            origin_host == host_hdr
            or origin in allowed
            or origin_host.startswith(("127.0.0.1:", "localhost:", "192.168."))
            or origin_host in ("127.0.0.1", "localhost")
        ):
            return True
        return False
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/monitor", "/ monitor.html", "/index.html"):
            idx_path = os.path.join(WEB_DIR, "index.html")
            self._serve_file(idx_path, "text/html; charset=utf-8")
            return

        if path.startswith("/web/"):
            rel = os.path.normpath(path[len("/web/") :]).lstrip("/\\")
            if ".." in rel:
                self._send_json({"error": {"message": "Forbidden"}}, 403)
                return
            self._serve_file(os.path.join(WEB_DIR, rel))
            return

        if path == "/favicon.ico":
            self._serve_file(os.path.join(WEB_DIR, "favicon.svg"), "image/svg+xml")
            return

        if path == "/health":
            alive = SUPERVISOR.proc is not None and SUPERVISOR.proc.poll() is None and SUPERVISOR.state == "ready"
            self._send_json(
                {
                    "ok": alive,
                    "status": "ok" if alive else SUPERVISOR.state,
                    "engine": "wzy-pp2-sm75",
                    "servable_by_prototype": True,
                    "ready": SUPERVISOR.ready_info,
                    "boot": SUPERVISOR.boot_info,
                },
                200 if alive else 503,
            )
            return

        if path in ("/status", "/v1/status"):
            st = dict(SUPERVISOR.current_stat)
            loaded = SUPERVISOR.state == "ready"
            self._send_json(
                {
                    "ok": True,
                    "loaded": loaded,
                    "state": SUPERVISOR.state,
                    "busy": SUPERVISOR.busy,
                    "queued": 0,
                    "phase": st.get("phase", "idle" if loaded else SUPERVISOR.state),
                    "prompt_tokens": st.get("prompt_tokens", 0),
                    "generated": st.get("generated", 0),
                    "max_tokens": st.get("max_tokens", 0),
                    "model": CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4"),
                    "activity": {"in_flight": 1 if SUPERVISOR.busy else 0},
                }
            )
            return

        if path == "/metrics":
            tot = SUPERVISOR.totals
            mean_dec_tps = (1000.0 * tot["output_tokens"] / tot["decode_ms"]) if tot["decode_ms"] > 0 else 0.0
            mean_pf_tps = (1000.0 * tot["read_tokens"] / tot["prefill_ms"]) if tot["prefill_ms"] > 0 else 0.0
            env = CONFIG.get("env", {})
            samp = CONFIG.get("sampling", {})
            self._send_json(
                {
                    "engine": SUPERVISOR.ready_info,
                    "engine_state": SUPERVISOR.state,
                    "boot": SUPERVISOR.boot_info,
                    "busy": SUPERVISOR.busy,
                    "live": SUPERVISOR.current_stat,
                    "last_prefill": SUPERVISOR.last_prefill,
                    "totals": dict(
                        tot,
                        decode_tok_s_mean=round(mean_dec_tps, 2),
                        prefill_tok_s_mean=round(mean_pf_tps, 1),
                    ),
                    "gpus": query_gpus(),
                    "sys": read_host_sys(),
                    "session_slots": SUPERVISOR.session_slots,
                    "config": {
                        "port": int(CONFIG.get("port", 11438)),
                        "mirror_ports": CONFIG.get("mirror_ports", []),
                        "launcher_port": int(CONFIG.get("launcher_port", 11439)),
                        "active_ports": PORT_MGR.active_ports() if PORT_MGR else [int(CONFIG.get("port", 11438))],
                        "auto_load": bool(CONFIG.get("auto_load", False)),
                        "split": int(env.get("WZY_SPLIT", 24)),
                        "ctx_max": int(env.get("WZY_CTX", 262144)),
                        "adapt_slots": int(env.get("WZY_ADAPT_SLOTS", 2048)),
                        "cpu_threads": int(env.get("WZY_CPU_THREADS", 14)),
                        "spec": int(samp.get("spec", 7)),
                        "spec_min_p": float(samp.get("spec_min_p", 0.72)),
                        "temperature": float(samp.get("temperature", 0.3)),
                        "top_p": float(samp.get("top_p", 0.75)),
                        "top_k": int(samp.get("top_k", 15)),
                        "min_p": float(samp.get("min_p", 0.02)),
                        "default_max_tokens": int(CONFIG.get("default_max_tokens", 512)),
                        "model_name": CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4"),
                    },
                    "series": list(SUPERVISOR.tps_series),
                    "requests": list(SUPERVISOR.recent_requests)[:40],
                }
            )
            return

        if path == "/api/requests":
            req_id = (qs.get("id") or [None])[0]
            if req_id:
                det = SUPERVISOR.request_details.get(req_id)
                if not det:
                    self._send_json({"error": {"message": f"Request {req_id} not found"}}, 404)
                    return
                self._send_json(det)
                return
            self._send_json(
                {
                    "loaded": SUPERVISOR.state == "ready",
                    "auto_load": bool(CONFIG.get("auto_load", False)),
                    "requests": list(SUPERVISOR.recent_requests)[:50],
                }
            )
            return

        if path == "/api/logs":
            self._send_json({"logs": list(SUPERVISOR.engine_logs), "state": SUPERVISOR.state, "boot": SUPERVISOR.boot_info})
            return

        if path in ("/v1/models", "/models"):
            names = [
                CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4"),
                "qwen3.8-flash-next-nvfp4-wzy",
                "strata",
            ]
            ctx = SUPERVISOR.ready_info.get("ctx_max", int(CONFIG.get("env", {}).get("WZY_CTX", 262144)))
            st_val = "loaded" if SUPERVISOR.state == "ready" else SUPERVISOR.state
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": m,
                            "object": "model",
                            "owned_by": "wzy-flash-serve",
                            "status": {"value": st_val},
                            "meta": {"n_ctx": ctx},
                        }
                        for m in names
                    ],
                }
            )
            return

        self._send_json({"error": {"message": f"Not found: {path}"}}, 404)

    def _apply_config_dict(self, body):
        env = CONFIG.setdefault("env", {})
        samp = CONFIG.setdefault("sampling", {})
        need_engine_restart = False

        if "split" in body and int(body["split"]) != int(env.get("WZY_SPLIT", 24)):
            env["WZY_SPLIT"] = str(max(8, min(40, int(body["split"]))))
            need_engine_restart = True
        if "ctx_max" in body and int(body["ctx_max"]) != int(env.get("WZY_CTX", 262144)):
            env["WZY_CTX"] = str(max(4096, min(262144, int(body["ctx_max"]))))
            need_engine_restart = True
        if "adapt_slots" in body and int(body["adapt_slots"]) != int(env.get("WZY_ADAPT_SLOTS", 2048)):
            env["WZY_ADAPT_SLOTS"] = str(max(0, min(4096, int(body["adapt_slots"]))))
            need_engine_restart = True
        if "cpu_threads" in body and int(body["cpu_threads"]) != int(env.get("WZY_CPU_THREADS", 14)):
            env["WZY_CPU_THREADS"] = str(max(1, min(32, int(body["cpu_threads"]))))
            need_engine_restart = True

        if "spec" in body:
            samp["spec"] = max(0, min(7, int(body["spec"])))
        if "spec_min_p" in body:
            samp["spec_min_p"] = max(0.05, min(0.99, float(body["spec_min_p"])))
        if "temperature" in body:
            samp["temperature"] = max(0.0, min(2.0, float(body["temperature"])))
        if "top_p" in body:
            samp["top_p"] = max(0.05, min(1.0, float(body["top_p"])))
        if "top_k" in body:
            samp["top_k"] = max(1, min(100, int(body["top_k"])))
        if "default_max_tokens" in body:
            CONFIG["default_max_tokens"] = max(16, min(65536, int(body["default_max_tokens"])))
        if "auto_load" in body:
            CONFIG["auto_load"] = bool(body["auto_load"])

        port_res = None
        if "port" in body or "mirror_ports" in body:
            new_port = int(body.get("port", CONFIG.get("port", 11438)))
            if "mirror_ports" in body:
                mp = body["mirror_ports"]
                if isinstance(mp, str):
                    mp = [int(x.strip()) for x in mp.split(",") if x.strip().isdigit()]
                elif isinstance(mp, list):
                    mp = [int(x) for x in mp if str(x).isdigit()]
                else:
                    mp = []
                CONFIG["mirror_ports"] = [p for p in mp if p != new_port]
            CONFIG["port"] = new_port
            if PORT_MGR is not None:
                port_res = PORT_MGR.sync_ports(
                    CONFIG["port"],
                    CONFIG.get("mirror_ports", []),
                    int(CONFIG.get("launcher_port", 11439)),
                )

        SUPERVISOR.save_config()
        return need_engine_restart, port_res

    def do_POST(self):
        if not self._check_auth():
            self._send_json({"error": {"message": "Unauthorized"}}, 401)
            return
        path = self.path.split("?", 1)[0]
        raw = self._read_body()
        try:
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except Exception as e:
            self._send_json({"error": {"message": f"Invalid JSON: {e}"}}, 400)
            return
        if path in (
            "/reset",
            "/api/launcher/start",
            "/load",
            "/api/launcher/stop",
            "/unload",
            "/api/launcher/restart",
            "/api/launcher/config",
            "/api/launcher/port",
            "/api/profile/save",
        ):
            if not self._check_admin_origin():
                self._send_json({"ok": False, "error": {"message": "Forbidden cross-origin management request"}}, 403)
                return
            client_ip = self.client_address[0] if self.client_address else "unknown"
            if SUPERVISOR is not None:
                SUPERVISOR._log(f"[wzy-audit] admin POST {path} from {client_ip}")

        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        force_flag = bool(body.get("force", False)) or (qs.get("force", ["0"])[0] in ("1", "true", "yes"))
        wait_val = body.get("wait", (qs.get("wait", [0.0])[0]))
        try:
            wait_s = float(wait_val or 0.0)
        except (TypeError, ValueError):
            wait_s = 0.0

        if path == "/reset":
            ok = SUPERVISOR.reset_state()
            self._send_json({"ok": ok})
            return
        if path == "/api/profile/save":
            ok = SUPERVISOR.save_profile()
            self._send_json({"ok": ok})
            return

        if path in ("/api/launcher/start", "/load"):
            if body:
                self._apply_config_dict(body)
            res = SUPERVISOR.start_async()
            res["port"] = int(CONFIG.get("port", 11438))
            res["active_ports"] = PORT_MGR.active_ports() if PORT_MGR else [int(CONFIG.get("port", 11438))]
            self._send_json(res)
            return

        if path in ("/api/launcher/stop", "/unload"):
            res = SUPERVISOR.stop_engine(force=force_flag, wait_s=wait_s)
            code = 409 if res.get("status") == "busy" else 200
            self._send_json(res, code)
            return

        if path == "/api/launcher/restart":
            if body:
                self._apply_config_dict(body)
            res = SUPERVISOR.restart_engine(force=force_flag, wait_s=wait_s)
            res["port"] = int(CONFIG.get("port", 11438))
            res["active_ports"] = PORT_MGR.active_ports() if PORT_MGR else [int(CONFIG.get("port", 11438))]
            code = 409 if res.get("status") == "busy" else 200
            self._send_json(res, code)
            return

        if path in ("/api/launcher/config", "/api/launcher/port"):
            need_restart, port_res = self._apply_config_dict(body)
            if port_res and not port_res.get("ok", True):
                self._send_json({"ok": False, "error": port_res.get("error", "端口切换失败"), "port_res": port_res}, 400)
                return
            restart_res = None
            if body.get("restart_if_needed") and need_restart and SUPERVISOR.state in ("ready", "booting"):
                restart_res = SUPERVISOR.restart_engine(force=force_flag, wait_s=wait_s)
                if restart_res and restart_res.get("status") == "busy":
                    self._send_json(restart_res, 409)
                    return
            self._send_json(
                {
                    "ok": True,
                    "need_engine_restart": need_restart,
                    "port": int(CONFIG.get("port", 11438)),
                    "active_ports": PORT_MGR.active_ports() if PORT_MGR else [int(CONFIG.get("port", 11438))],
                    "port_res": port_res,
                }
            )
            return

        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat(body)
        elif path in ("/v1/messages", "/messages"):
            self._handle_messages(body)
        elif path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            self._handle_count_tokens(body)
        elif path in ("/v1/responses", "/responses"):
            self._handle_responses(body)
        elif path in ("/v1/completions", "/completions"):
            self._handle_completions(body)
        else:
            self._send_json({"error": {"message": f"Not found: {path}"}}, 404)

    def _extract_params(self, body, default_max_key="max_tokens"):
        samp = CONFIG.get("sampling", {})
        stop_ids = [tk.ID_IM_END, tk.ID_ENDOFTEXT]
        stop_strs = body.get("stop") or body.get("stop_sequences")
        if isinstance(stop_strs, str):
            stop_strs = [stop_strs]
        elif not isinstance(stop_strs, list):
            stop_strs = []
        max_t = (
            body.get(default_max_key)
            or body.get("max_tokens")
            or body.get("max_completion_tokens")
            or body.get("max_output_tokens")
            or CONFIG.get("default_max_tokens", 2048)
        )
        return {
            "max_tokens": int(max_t),
            "temperature": float(body.get("temperature", samp.get("temperature", 0.3))),
            "top_p": float(body.get("top_p", samp.get("top_p", 0.75))),
            "top_k": int(body.get("top_k", samp.get("top_k", 15))),
            "min_p": float(body.get("min_p", samp.get("min_p", 0.02))),
            "repetition_penalty": float(body.get("repetition_penalty", samp.get("repetition_penalty", 1.0))),
            "spec": int(body.get("spec", samp.get("spec", 7))),
            "spec_min_p": float(body.get("spec_min_p", samp.get("spec_min_p", 0.72))),
            "no_reuse": bool(body.get("no_reuse", False)),
            "stop_ids": stop_ids,
            "stop_strs": [s for s in stop_strs if s],
        }

    @staticmethod
    def _build_preview(messages):
        try:
            lines = []
            for m in messages[-4:]:
                r = m.get("role", "user")
                c = tk._render_content(m.get("content"))
                tcs = m.get("tool_calls") or []
                if tcs:
                    tc_txt = ", ".join(
                        f"{tc.get('function', tc).get('name')}({json.dumps(tc.get('function', tc).get('arguments', {}), ensure_ascii=False)[:160]})"
                        for tc in tcs
                    )
                    c = (c + "\n[tool_calls: " + tc_txt + "]").strip()
                lines.append(f"[{r}] {(c or '')[:1200]}")
            return "\n\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    def _resolve_tools_and_choice(tools, tool_choice):
        tchoice_kind, tchoice_name = fe.tool_choice_of(tool_choice)
        if tool_choice is not None and tchoice_kind == "unknown":
            raise ValueError(f"Unsupported tool_choice: {tool_choice!r}")
        if tchoice_kind == "none":
            return None, None
        tool_list = list(tools) if tools else None
        names = {t.get("name") for t in (tool_list or []) if isinstance(t, dict) and t.get("name")}
        if tchoice_kind == "required" and not names:
            raise ValueError("tool_choice requires a tool call, but no tools were provided")
        if tchoice_kind == "named":
            if not names or tchoice_name not in names:
                raise ValueError(f"Tool {tchoice_name!r} requested in tool_choice is not in tools")
        force = fe.forced_call(tool_choice, tool_list) if tool_list else None
        return tool_list, force

    def _handle_chat(self, body):
        try:
            messages, tools, kw = fe.openai_to_messages(body)
            active_tools, force = self._resolve_tools_and_choice(tools, body.get("tool_choice"))
        except ValueError as e:
            self._send_json({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
            return
        if not messages:
            self._send_json({"error": {"message": "messages required", "type": "invalid_request_error"}}, 400)
            return
        if "enable_thinking" in body and body["enable_thinking"] is False:
            kw["enable_thinking"] = False

        enable_thinking = kw.get("enable_thinking", True) is not False

        try:
            emb_info = {}
            messages, img_sources = SUPERVISOR.prepare_vision(messages)
            ids = SUPERVISOR.build_chat_ids(messages, active_tools, kw, force=force, allow_cache=not img_sources)
            ids, emb_info = SUPERVISOR.vision_ids(ids, img_sources)
        except Exception as e:
            self._send_json({"error": {"message": f"Template error: {e}", "type": "invalid_request_error"}}, 400)
            return
        params = self._extract_params(body)
        params.update(emb_info)
        model_id = body.get("model") or CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4")
        cid = "chatcmpl-" + uuid.uuid4().hex[:20]
        created = int(time.time())
        stream = bool(body.get("stream", False))
        meta = {
            "id": cid,
            "path": "/v1/chat/completions",
            "stream": stream,
            "input_preview": self._build_preview(messages),
        }

        def make_chunk(delta, finish=None):
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        def generate_openai_chunks():
            yield make_chunk({"role": "assistant", "content": ""})
            calls = 0
            streamed = {}
            finished = set()
            completed_calls = []
            think_parts = []
            ans_parts = []
            for kind, payload in SUPERVISOR.run_parsed_events(
                ids, params, enable_thinking, active_tools, force=force, meta=meta
            ):
                if kind == "prefill":
                    continue
                if kind == "event":
                    ev = payload
                    if ev.kind == "reasoning" and ev.text:
                        think_parts.append(ev.text)
                        yield make_chunk({"reasoning_content": ev.text})
                    elif ev.kind == "content" and ev.text:
                        ans_parts.append(ev.text)
                        yield make_chunk({"content": ev.text})
                    elif ev.kind == "tool_start":
                        streamed[ev.call.id] = calls
                        calls += 1
                        yield make_chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": streamed[ev.call.id],
                                        "id": ev.call.id,
                                        "type": "function",
                                        "function": {"name": ev.call.name, "arguments": ""},
                                    }
                                ]
                            }
                        )
                    elif ev.kind == "tool_args":
                        yield make_chunk(
                            {
                                "tool_calls": [
                                    {"index": streamed[ev.call.id], "function": {"arguments": ev.text}}
                                ]
                            }
                        )
                    elif ev.kind == "tool_call" and ev.call.id in streamed:
                        finished.add(ev.call.id)
                        completed_calls.append(ev.call)
                    elif ev.kind == "tool_call":
                        completed_calls.append(ev.call)
                        yield make_chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": calls,
                                        "id": ev.call.id,
                                        "type": "function",
                                        "function": {
                                            "name": ev.call.name,
                                            "arguments": json.dumps(ev.call.arguments, ensure_ascii=False),
                                        },
                                    }
                                ]
                            }
                        )
                        calls += 1
                elif kind == "done":
                    dn, pf_info, raw_gen_ids = payload
                    raw_fin = dn.get("finish", "stop")
                    whole = bool(calls and streamed.keys() <= finished)
                    finish = "tool_calls" if (whole and raw_fin == "stop") else ("stop" if raw_fin == "cancelled" else raw_fin)
                    p_tok = dn.get("prompt_tokens", len(ids))
                    c_tok = dn.get("generated", 0)
                    reused = dn.get("reused", 0)
                    p_ms = dn.get("prompt_ms", pf_info.get("prompt_ms", 0.0))
                    d_ms = dn.get("decode_ms", 0.0)
                    ans_text = "".join(ans_parts)
                    rc_text = "".join(think_parts)
                    insp_out = ans_text
                    if completed_calls:
                        tc_fmt = "\n".join(
                            f"[tool_call #{i+1}] {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
                            for i, c in enumerate(completed_calls)
                        )
                        insp_out = (insp_out + "\n\n" + tc_fmt).strip()
                    SUPERVISOR.attach_request_text(cid, rc_text, insp_out)
                    if raw_fin != "cancelled":
                        SUPERVISOR.record_conv_ids(messages, active_tools, ans_text, completed_calls, list(ids) + raw_gen_ids)
                    last = make_chunk({}, finish)
                    last["usage"] = {
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "total_tokens": p_tok + c_tok,
                        "prompt_tokens_details": {"cached_tokens": reused},
                    }
                    last["timings"] = {
                        "cache_n": reused,
                        "prompt_n": p_tok - reused,
                        "prompt_ms": p_ms,
                        "prompt_per_second": round(1000.0 * max(1, p_tok - reused) / p_ms, 2) if p_ms > 0 else 0.0,
                        "predicted_n": c_tok,
                        "predicted_ms": d_ms,
                        "predicted_per_second": dn.get("decode_tps", 0.0),
                        "draft_n": dn.get("draft_n", 0),
                        "draft_accepted": dn.get("draft_accepted", 0),
                        "vram_hit": dn.get("vram_hit", 1.0),
                        "promoted": dn.get("promoted", 0),
                    }
                    yield last

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            try:
                for chk in generate_openai_chunks():
                    raw_chk = f"data: {json.dumps(chk, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(raw_chk)
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                try:
                    err_chk = f"data: {json.dumps({'error': {'message': str(e)}}, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(err_chk)
                    self.wfile.flush()
                except Exception:
                    pass
            return

        # Non-streaming: collect chunks into standard OpenAI chat.completion object
        try:
            buffered = list(generate_openai_chunks())
        except Exception as e:
            self._send_json({"error": {"message": str(e)}}, 503)
            return

        content_acc, reasoning_acc, by_index, last_chk = [], [], {}, buffered[-1]
        for c in buffered:
            d = c["choices"][0]["delta"]
            content_acc.append(d.get("content") or "")
            reasoning_acc.append(d.get("reasoning_content") or "")
            for tc in d.get("tool_calls") or []:
                cur = by_index.setdefault(
                    tc.get("index", len(by_index)),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                cur["id"] = tc.get("id") or cur["id"]
                fn = tc.get("function") or {}
                cur["function"]["name"] += fn.get("name") or ""
                cur["function"]["arguments"] += fn.get("arguments") or ""
        calls_out = [by_index[i] for i in sorted(by_index)]
        fin_reason = last_chk["choices"][0]["finish_reason"]
        if fin_reason != "tool_calls":
            valid_calls = []
            for c in calls_out:
                try:
                    json.loads(c["function"]["arguments"])
                    valid_calls.append(c)
                except Exception:
                    pass
            calls_out = valid_calls
        msg_out = {"role": "assistant", "content": "".join(content_acc) or (None if calls_out else "")}
        if "".join(reasoning_acc):
            msg_out["reasoning_content"] = "".join(reasoning_acc)
        if calls_out:
            msg_out["tool_calls"] = calls_out
        resp_obj = {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "message": msg_out, "finish_reason": fin_reason}],
            "usage": last_chk.get("usage", {}),
            "timings": last_chk.get("timings", {}),
        }
        self._send_json(resp_obj)

    def _handle_count_tokens(self, body):
        try:
            messages, tools, kw = fe.anthropic_to_messages(body)
            ids = tk.render_and_encode(SUPERVISOR.normalize_messages(messages), tools=tools, add_generation_prompt=True, **kw)
            self._send_json({"input_tokens": len(ids)})
        except Exception as e:
            self._send_json({"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}, 400)

    def _handle_messages(self, body):
        """Anthropic `/v1/messages` endpoint (used by ZCode / Claude Code / Cline in Anthropic mode)."""
        try:
            messages, tools, kw = fe.anthropic_to_messages(body)
            active_tools, force = self._resolve_tools_and_choice(tools, body.get("tool_choice"))
        except ValueError as e:
            self._send_json({"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}, 400)
            return
        if not messages:
            self._send_json({"type": "error", "error": {"type": "invalid_request_error", "message": "messages required"}}, 400)
            return

        enable_thinking = kw.get("enable_thinking", True) is not False

        try:
            emb_info = {}
            messages, img_sources = SUPERVISOR.prepare_vision(messages)
            ids = SUPERVISOR.build_chat_ids(messages, active_tools, kw, force=force, allow_cache=not img_sources)
            ids, emb_info = SUPERVISOR.vision_ids(ids, img_sources)
        except Exception as e:
            self._send_json({"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}, 400)
            return

        params = self._extract_params(body)
        model_id = body.get("model") or CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4")
        mid = "msg_" + uuid.uuid4().hex[:24]
        params.update(emb_info)
        stream = bool(body.get("stream", False))
        meta = {
            "id": mid,
            "path": "/v1/messages",
            "stream": stream,
            "input_preview": self._build_preview(messages),
        }

        def generate_anthropic_events():
            yield (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": mid,
                        "type": "message",
                        "role": "assistant",
                        "model": model_id,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": len(ids), "output_tokens": 0},
                    },
                },
            )
            index = -1
            open_kind = None
            used_tool = False
            streamed = set()
            finished = set()
            completed_calls = []
            think_parts = []
            ans_parts = []

            def close_blk():
                return ("content_block_stop", {"type": "content_block_stop", "index": index})

            for kind, payload in SUPERVISOR.run_parsed_events(
                ids, params, enable_thinking, active_tools, force=force, meta=meta
            ):
                if kind == "prefill":
                    continue
                if kind == "event":
                    ev = payload
                    if ev.kind == "tool_args":
                        yield (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "input_json_delta", "partial_json": ev.text},
                            },
                        )
                        continue
                    if ev.kind == "tool_call" and ev.call.id in streamed:
                        finished.add(ev.call.id)
                        completed_calls.append(ev.call)
                        continue
                    want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
                    if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                        continue
                    if ev.kind == "tool_start":
                        streamed.add(ev.call.id)
                        used_tool = True
                    if open_kind != want or want == "tool_use":
                        if open_kind is not None:
                            yield close_blk()
                        index += 1
                        open_kind = want
                        blk = {
                            "thinking": {"type": "thinking", "thinking": "", "signature": ""},
                            "text": {"type": "text", "text": ""},
                            "tool_use": {
                                "type": "tool_use",
                                "id": ev.call.id if ev.call else "",
                                "name": ev.call.name if ev.call else "",
                                "input": {},
                            },
                        }[want]
                        yield ("content_block_start", {"type": "content_block_start", "index": index, "content_block": blk})
                    if want == "thinking":
                        think_parts.append(ev.text)
                        yield (
                            "content_block_delta",
                            {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": ev.text}},
                        )
                    elif want == "text":
                        ans_parts.append(ev.text)
                        yield (
                            "content_block_delta",
                            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": ev.text}},
                        )
                    elif ev.kind == "tool_call":
                        used_tool = True
                        completed_calls.append(ev.call)
                        yield (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False),
                                },
                            },
                        )
                elif kind == "done":
                    dn, _, raw_gen_ids = payload
                    if open_kind is not None:
                        yield close_blk()
                    raw_fin = dn.get("finish", "stop")
                    stop_reason = (
                        "tool_use"
                        if (used_tool and streamed <= finished and raw_fin == "stop")
                        else {"stop": "end_turn", "length": "max_tokens", "cancelled": "end_turn"}.get(raw_fin, "end_turn")
                    )
                    ans_text = "".join(ans_parts)
                    rc_text = "".join(think_parts)
                    insp_out = ans_text
                    if completed_calls:
                        tc_fmt = "\n".join(
                            f"[tool_use #{i+1}] {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
                            for i, c in enumerate(completed_calls)
                        )
                        insp_out = (insp_out + "\n\n" + tc_fmt).strip()
                    SUPERVISOR.attach_request_text(mid, rc_text, insp_out)
                    if raw_fin != "cancelled":
                        SUPERVISOR.record_conv_ids(messages, active_tools, ans_text, completed_calls, list(ids) + raw_gen_ids)
                    reused = max(0, min(dn.get("reused", 0), len(ids) - 1))
                    yield (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                            "usage": {
                                "input_tokens": len(ids) - reused,
                                "cache_read_input_tokens": reused,
                                "output_tokens": dn.get("generated", 0),
                            },
                        },
                    )
                    yield ("message_stop", {"type": "message_stop"})

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            try:
                for ev_name, ev_obj in generate_anthropic_events():
                    pkt = f"event: {ev_name}\ndata: {json.dumps(ev_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(pkt)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        try:
            evs = list(generate_anthropic_events())
        except Exception as e:
            self._send_json({"type": "error", "error": {"type": "api_error", "message": str(e)}}, 503)
            return

        msg_res, blocks = None, []
        for name, e in evs:
            if name == "message_start":
                msg_res = dict(e["message"])
            elif name == "content_block_start":
                blocks.append(dict(e["content_block"]))
            elif name == "content_block_delta":
                d, b = e["delta"], blocks[-1]
                if d["type"] == "text_delta":
                    b["text"] += d["text"]
                elif d["type"] == "thinking_delta":
                    b["thinking"] += d["thinking"]
                else:
                    b["_json"] = b.get("_json", "") + d["partial_json"]
            elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
                b = blocks[-1]
                try:
                    b["input"] = json.loads(b.pop("_json") or "{}")
                except ValueError:
                    blocks.pop()
            elif name == "message_delta":
                msg_res["stop_reason"] = e["delta"]["stop_reason"]
                msg_res["stop_sequence"] = e["delta"].get("stop_sequence")
                msg_res["usage"].update(e["usage"])
        msg_res["content"] = blocks
        self._send_json(msg_res)

    def _handle_responses(self, body):
        """OpenAI `/v1/responses` endpoint (used by Codex CLI / Responses-mode Agent Harnesses)."""
        model_id = body.get("model") or CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4")
        try:
            title_evs = rp.thread_title_events(body, model_id)
            if title_evs is not None:
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self._cors()
                    self.end_headers()
                    for ev_obj in title_evs:
                        self.wfile.write(f"event: {ev_obj['type']}\ndata: {json.dumps(ev_obj, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                else:
                    self._send_json(rp.collect(title_evs))
                return
            rp.check_request(body)
            messages = rp.input_messages(body)
            tools, tool_names, _ = rp.request_tools(body)
            ptools = rp.prompt_tools(body, tools)
            kw = rp.template_kwargs(body, {})
        except rp.ResponsesError as e:
            self._send_json(e.body(), e.status)
            return
        except ValueError as e:
            self._send_json(rp.error_body(str(e)), 400)
            return

        enable_thinking = kw.get("enable_thinking", True) is not False
        force = fe.forced_call(body.get("tool_choice"), tools)
        try:
            emb_info = {}
            messages, img_sources = SUPERVISOR.prepare_vision(messages)
            ids = SUPERVISOR.build_chat_ids(messages, ptools, kw, force=force, allow_cache=not img_sources)
            ids, emb_info = SUPERVISOR.vision_ids(ids, img_sources)
            rp.prompt_made(body, ptools)
        except Exception as e:
            self._send_json(rp.error_body(str(e)), 400)
            return

        params = self._extract_params(body, default_max_key="max_output_tokens")
        params.update(emb_info)
        stream = bool(body.get("stream", False))
        include_enc = "reasoning.encrypted_content" in (body.get("include") or [])
        asm = rp.Assembler(body, model_id, len(ids), tool_names, include_enc, json_mode=False)
        rid = asm.response["id"]
        meta = {
            "id": rid,
            "path": "/v1/responses",
            "stream": stream,
            "input_preview": self._build_preview(messages),
        }

        def generate_resp_events():
            for ev_obj in asm.start():
                yield ev_obj
            completed_calls = []
            think_parts = []
            ans_parts = []
            for kind, payload in SUPERVISOR.run_parsed_events(
                ids, params, enable_thinking, tools, force=force, meta=meta
            ):
                if kind == "prefill":
                    continue
                if kind == "event":
                    ev = payload
                    if ev.kind == "reasoning" and ev.text:
                        think_parts.append(ev.text)
                    elif ev.kind == "content" and ev.text:
                        ans_parts.append(ev.text)
                    elif ev.kind == "tool_call":
                        completed_calls.append(ev.call)
                    for ev_obj in asm.feed(ev):
                        yield ev_obj
                elif kind == "done":
                    dn, _, raw_gen_ids = payload
                    ans_text = "".join(ans_parts)
                    rc_text = "".join(think_parts)
                    insp_out = ans_text
                    if completed_calls:
                        tc_fmt = "\n".join(
                            f"[function_call #{i+1}] {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
                            for i, c in enumerate(completed_calls)
                        )
                        insp_out = (insp_out + "\n\n" + tc_fmt).strip()
                    SUPERVISOR.attach_request_text(rid, rc_text, insp_out)
                    if dn.get("finish") != "cancelled":
                        SUPERVISOR.record_conv_ids(messages, ptools, ans_text, completed_calls, list(ids) + raw_gen_ids)
                    done_summary = {
                        "finish": dn.get("finish", "stop"),
                        "completion_tokens": dn.get("generated", 0),
                        "prompt_tokens": dn.get("prompt_tokens", len(ids)),
                        "reused": dn.get("reused", 0),
                    }
                    for ev_obj in asm.finish(done_summary):
                        yield ev_obj

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            try:
                for ev_obj in generate_resp_events():
                    pkt = f"event: {ev_obj['type']}\ndata: {json.dumps(ev_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(pkt)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        try:
            evs = list(generate_resp_events())
            self._send_json(rp.collect(evs))
        except Exception as e:
            self._send_json(rp.error_body(str(e), "server_error"), 503)

    def _handle_completions(self, body):
        prompt = body.get("prompt") or ""
        ids = tk.encode(prompt, allow_special=True)
        params = self._extract_params(body)
        model_id = body.get("model") or CONFIG.get("model_name", "qwen3.8-flash-next-gptq-nvfp4")
        cid = "cmpl-" + uuid.uuid4().hex[:20]
        created = int(time.time())
        meta = {"id": cid, "path": "/v1/completions", "stream": False, "input_preview": str(prompt)[:2000]}
        out_ids, done_info = [], {}
        try:
            for ev, data in SUPERVISOR.generate_stream(ids, params, meta=meta):
                if ev == "token":
                    out_ids.append(data)
                elif ev == "done":
                    done_info = data
        except Exception as e:
            self._send_json({"error": {"message": str(e)}}, 503)
            return
        text = tk.decode(out_ids, skip_special=True)
        resp_obj = {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": done_info.get("finish", "stop")}],
            "usage": {
                "prompt_tokens": done_info.get("prompt_tokens", len(ids)),
                "completion_tokens": len(out_ids),
                "total_tokens": done_info.get("prompt_tokens", len(ids)) + len(out_ids),
            },
        }
        SUPERVISOR.attach_request_text(cid, "", text, resp_obj)
        self._send_json(resp_obj)

def main():
    global SUPERVISOR, PORT_MGR, CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--launcher-port", type=int, default=0)
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        CONFIG = json.load(f)
    if args.port > 0:
        CONFIG["port"] = args.port
    if args.launcher_port > 0:
        CONFIG["launcher_port"] = args.launcher_port
    CONFIG.setdefault("launcher_port", 11439)

    threading.Thread(target=tk.load_vocab, daemon=True).start()

    host = CONFIG.get("host", "0.0.0.0")
    primary_port = int(CONFIG.get("port", 11438))
    mirror_ports = [int(p) for p in CONFIG.get("mirror_ports", []) if int(p) != primary_port]
    launcher_port = int(CONFIG.get("launcher_port", 11439))

    PORT_MGR = PortManager(host, LjqHandler)
    PORT_MGR.sync_ports(primary_port, mirror_ports, launcher_port)

    SUPERVISOR = EngineSupervisor(CONFIG, cfg_path)

    stop_ev = threading.Event()

    def _sig_handler(signum, frame):
        print(f"[wzy-serve] caught signal {signum}, shutting down...", flush=True)
        SUPERVISOR.shutdown()
        PORT_MGR.shutdown_all()
        stop_ev.set()
        os._exit(0)

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    while not stop_ev.is_set():
        time.sleep(1.0)


if __name__ == "__main__":
    main()
