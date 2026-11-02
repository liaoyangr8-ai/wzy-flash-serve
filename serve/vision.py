# -*- coding: utf-8 -*-
"""vision.py -- image -> text-model embeddings for wzy-flash-serve.

The image half is Strata's CPU reader (`strata-vision`, built on llama.cpp's mtmd + the model's
`mmproj-*.gguf`): it prints `READY <n_embd>` and then answers `ENC <image> <out>` with
`OK <n_tokens> <nx> <ny> <ms>`, writing `int32 {SVE1, n_tokens, nx, ny, n_embd}` followed by
`float32 [n_tokens][n_embd]`.  Those rows already live in the text model's embedding space: the
engine drops them onto the `<|image_pad|>` tokens of the prompt (see `wzy_engine_v2.cu`,
`inject_img_emb`).  Nothing of the picture ever touches a GPU here, so the 24 GB of experts are
untouched while a picture is read.
"""
import base64
import hashlib
import io
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
MARKER = VISION_START + IMAGE_PAD + VISION_END
SVE_MAGIC = 0x31455653


#: What the server leaves in the message text where the picture was.  It cannot be the marker itself: the chat
#: template protects literal `<|...|>` runs (so an agent quoting these docs is not tokenized as control tokens),
#: which would turn the marker into ordinary text.  A private-use sentinel survives the template untouched and is
#: swapped for the real `<|vision_start|> pad×n <|vision_end|>` ids after encoding.
SENTINEL = "\uf8f0\uf8f1"

class VisionError(RuntimeError):
    """A picture could not be read (bad bytes, encoder down, ...). The request is refused."""


class Vision:
    def __init__(self, cfg):
        cfg = cfg or {}
        self.exe = cfg.get("exe") or ""
        self.mmproj = cfg.get("mmproj") or ""
        self.model = cfg.get("model") or ""
        self.threads = int(cfg.get("threads", 8) or 8)
        self.gpu = bool(cfg.get("gpu", False))
        self.max_tokens = int(cfg.get("max_tokens", 0) or 0)
        self.min_tokens = int(cfg.get("min_tokens", 0) or 0)
        self.dir = Path(cfg.get("dir") or "/tmp/wzy-vision")
        self.ready_s = float(cfg.get("ready_s", 300))
        self.encode_s = float(cfg.get("encode_s", 300))
        self.cache = {}
        self.cache_max = 64
        self.proc = None
        self.lines = None
        self.n_embd = 0
        self.lock = threading.Lock()

    # ---- availability -------------------------------------------------------------------
    @property
    def enabled(self):
        return bool(self.exe and self.mmproj and self.model and os.path.isfile(self.exe))

    def info(self):
        return {"enabled": self.enabled, "exe": self.exe, "mmproj": self.mmproj,
                "n_embd": self.n_embd, "running": self.proc is not None and self.proc.poll() is None}

    # ---- encoder process ----------------------------------------------------------------
    def _start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        argv = [self.exe, "--mmproj", self.mmproj, "--model", self.model, "--threads", str(self.threads)]
        if self.gpu:
            argv.append("--gpu")
        if self.max_tokens:
            argv += ["--max-tokens", str(self.max_tokens)]
        if self.min_tokens:
            argv += ["--min-tokens", str(self.min_tokens)]
        # cwd = the picture directory: `ENC <img> <out>` names are relative to it (the reader's own convention).
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=str(self.dir))
        lines = self.lines = queue.Queue()
        proc = self.proc
        def read_output():
            try:
                for line in proc.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        threading.Thread(target=read_output, daemon=True).start()
        deadline = time.monotonic() + self.ready_s
        while True:
            line = self._read_line(deadline, "the vision encoder did not start in %.1fs" % self.ready_s)
            if line.startswith("READY"):
                self.n_embd = int(line.split()[1])
                return

    def _read_line(self, deadline, timeout_message):
        try:
            line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            self._kill()
            raise VisionError(timeout_message)
        if line is None:
            self._kill()
            raise VisionError("the vision encoder stopped")
        return line.strip()

    def _kill(self):
        proc, self.proc = self.proc, None
        self.lines = None
        if proc is not None:
            proc.kill()
            proc.wait()

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass

    # ---- pictures -----------------------------------------------------------------------
    def _png(self, data):
        """Any picture -> PNG bytes, so the encoder's stb decoder always sees a format it reads."""
        try:
            from PIL import Image, ImageOps
        except Exception:
            return data
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            im = ImageOps.exif_transpose(im)
        except Exception as exc:
            raise VisionError("the image could not be read (%s)" % exc)
        if im.mode == "RGBA" or im.mode in ("LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    @staticmethod
    def decode_source(source):
        """`data:` URI / http(s) URL / local path / raw bytes -> bytes."""
        if isinstance(source, (bytes, bytearray)):
            return bytes(source)
        s = (source or "").strip()
        if not s:
            raise VisionError("empty image source")
        if s.startswith("data:"):
            head, _, payload = s.partition(",")
            if ";base64" in head:
                return base64.b64decode(payload, validate=False)
            return urllib.parse.unquote_to_bytes(payload)
        if s.startswith(("http://", "https://")):
            with urllib.request.urlopen(s, timeout=60) as r:
                return r.read()
        if s.startswith("file://"):
            s = s[7:]
        if os.path.isfile(s):
            with open(s, "rb") as f:
                return f.read()
        raise VisionError("the image source is neither a data URI, a URL nor a local file")

    def encode(self, data):
        """PNG bytes -> (sve_path, n_tokens).  Cached by content hash."""
        png = self._png(data)
        key = hashlib.sha256(png).hexdigest()[:32]
        with self.lock:
            hit = self.cache.get(key)
            if hit and os.path.isfile(hit[0]):
                self.cache[key] = self.cache.pop(key)
                return hit
            self._start()
            self.dir.mkdir(parents=True, exist_ok=True)
            img, out = self.dir / ("%s.img" % key), self.dir / ("%s.sve" % key)
            img.write_bytes(png)
            try:
                self.proc.stdin.write("ENC %s %s\n" % (img.name, out.name))
                self.proc.stdin.flush()
                deadline = time.monotonic() + self.encode_s
                while True:
                    line = self._read_line(deadline, "the vision encoder timed out")
                    if line.startswith("OK "):
                        parts = line.split()
                        n_tok = int(parts[1])
                        self.cache[key] = (out, n_tok)
                        while len(self.cache) > self.cache_max:
                            old = next(iter(self.cache))
                            if old == key:
                                break
                            p = self.cache.pop(old)[0]
                            try:
                                p.unlink(missing_ok=True)
                            except Exception:
                                pass
                        return out, n_tok
                    if line.startswith("ERR"):
                        raise VisionError("the image could not be read: " + line[3:].strip())
            finally:
                try:
                    img.unlink(missing_ok=True)
                except Exception:
                    pass

    def combined(self, encoded, key):
        """Concatenate per-image SVE rows into one SVE file: {SVE1, total, total, 1, n_embd}.

        The engine reads one file per request and one (start, count) run per image, so the rows
        must be in prompt order -- `encoded` is.
        """
        total = sum(n for _, n in encoded)
        dim = 0
        for path, n in encoded:
            with open(path, "rb") as f:
                hdr = struct.unpack("5i", f.read(20))
            if hdr[0] != SVE_MAGIC:
                raise VisionError("bad vision embeddings file")
            dim = hdr[4]
        out_path = self.dir / ("%s.sve" % key)
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as dst:
            dst.write(struct.pack("5i", SVE_MAGIC, total, total, 1, dim))
            for path, n in encoded:
                with open(path, "rb") as f:
                    f.read(20)
                    remain = n * dim * 4
                    while remain > 0:
                        chunk = f.read(min(remain, 1 << 20))
                        if not chunk:
                            raise VisionError("short vision embeddings file")
                        dst.write(chunk)
                        remain -= len(chunk)
        os.replace(tmp, out_path)
        return out_path, total
