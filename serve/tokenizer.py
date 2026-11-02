# -*- coding: utf-8 -*-
"""Byte-level BPE tokenizer + Qwen3.8-Flash-Next ChatML formatter + UTF-8 streaming decoder."""
import functools
import json
import os
import re
import unicodedata

try:
    import regex as _re_mod
except ImportError:
    _re_mod = None

DEFAULT_TOK_DIR = "/home/liaoyang/models/Maximilian228--Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata/pack/tokenizer"

SPECIAL_TOKENS = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|object_ref_start|>": 248047,
    "<|object_ref_end|>": 248048,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    "<|quad_start|>": 248051,
    "<|quad_end|>": 248052,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|vision_pad|>": 248055,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
    "<tool_call>": 248058,
    "</tool_call>": 248059,
    "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061,
    "<|fim_suffix|>": 248062,
    "<|fim_pad|>": 248063,
    "<|repo_name|>": 248064,
    "<|file_sep|>": 248065,
    "<tool_response>": 248066,
    "</tool_response>": 248067,
    "<think>": 248068,
    "</think>": 248069,
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}
SEMANTIC_TAG_IDS = {248058, 248059, 248066, 248067, 248068, 248069}

ID_ENDOFTEXT = 248044
ID_IM_START = 248045
ID_IM_END = 248046
ID_TOOL_CALL_START = 248058
ID_TOOL_CALL_END = 248059
ID_THINK_START = 248068
ID_THINK_END = 248069
_SPECIAL_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(SPECIAL_TOKENS.keys(), key=len, reverse=True)) + ")"
)

_PRE_PATTERN_STR = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_PRE_REGEX = _re_mod.compile(_PRE_PATTERN_STR) if _re_mod is not None else None


def _byte_encoder():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENC = _byte_encoder()
BYTE_DEC = {v: k for k, v in BYTE_ENC.items()}


@functools.lru_cache(maxsize=1)
def load_vocab(tok_dir=DEFAULT_TOK_DIR):
    vocab = json.load(open(os.path.join(tok_dir, "vocab.json"), encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}
    ranks = {}
    with open(os.path.join(tok_dir, "merges.txt"), encoding="utf-8") as f:
        idx = 0
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 2:
                ranks[(parts[0], parts[1])] = idx
                idx += 1
    return vocab, inv, ranks


def _fallback_split(text):
    _contr = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")
    out = []
    i, n = 0, len(text)
    while i < n:
        hit = None
        for c in _contr:
            if text[i : i + len(c)].lower() == c:
                hit = text[i : i + len(c)]
                break
        if hit is not None:
            out.append(hit)
            i += len(hit)
            continue
        j = i
        if (
            j < n
            and text[j] not in "\r\n"
            and unicodedata.category(text[j])[0] not in ("L", "N")
            and j + 1 < n
            and unicodedata.category(text[j + 1])[0] in ("L", "M")
        ):
            j += 1
        if j < n and unicodedata.category(text[j])[0] in ("L", "M"):
            k = j
            while k < n and unicodedata.category(text[k])[0] in ("L", "M"):
                k += 1
            out.append(text[i:k])
            i = k
            continue
        if unicodedata.category(text[i])[0] == "N":
            out.append(text[i])
            i += 1
            continue
        j = i
        if text[j] == " ":
            j += 1
        if j < n and not text[j].isspace() and unicodedata.category(text[j])[0] not in ("L", "M", "N"):
            k = j
            while k < n and not text[k].isspace() and unicodedata.category(text[k])[0] not in ("L", "M", "N"):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        j = i
        while j < n and text[j].isspace() and text[j] not in "\r\n":
            j += 1
        if j < n and text[j] in "\r\n":
            k = j
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        if text[i].isspace():
            k = i
            while k < n and text[k].isspace():
                k += 1
            out.append(text[i:k])
            i = k
            continue
        out.append(text[i])
        i += 1
    return out


@functools.lru_cache(maxsize=65536)
def _bpe_piece(chunk):
    vocab, _, ranks = load_vocab()
    sym = list(chunk)
    while len(sym) > 1:
        best, bi = None, None
        for i in range(len(sym) - 1):
            r = ranks.get((sym[i], sym[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if bi is None:
            break
        sym = sym[:bi] + [sym[bi] + sym[bi + 1]] + sym[bi + 2 :]
    return tuple(vocab[s] for s in sym)


def encode(text, allow_special=True):
    if not text:
        return []
    load_vocab()
    ids = []
    segments = _SPECIAL_RE.split(text) if allow_special else [text]
    for seg in segments:
        if not seg:
            continue
        if allow_special and seg in SPECIAL_TOKENS:
            ids.append(SPECIAL_TOKENS[seg])
            continue
        pieces = _PRE_REGEX.findall(seg) if _PRE_REGEX is not None else _fallback_split(seg)
        for p in pieces:
            chunk = "".join(BYTE_ENC[b] for b in p.encode("utf-8"))
            ids.extend(_bpe_piece(chunk))
    return ids
def encode_with_plain_spans(text, plain_spans=None):
    """Encodes `text` with `allow_special=True` outside `plain_spans`, and `allow_special=False` inside `plain_spans`."""
    if not text:
        return []
    if not plain_spans:
        return encode(text, allow_special=True)
    ids = []
    pos = 0
    for s, e in plain_spans:
        if s > pos:
            ids.extend(encode(text[pos:s], allow_special=True))
        if e > s:
            ids.extend(encode(text[s:e], allow_special=False))
        pos = e
    if pos < len(text):
        ids.extend(encode(text[pos:], allow_special=True))
    return ids


def token_to_bytes(tid, skip_special=True, keep_tool_tags=False):
    if keep_tool_tags and tid in SEMANTIC_TAG_IDS:
        return ID_TO_SPECIAL[tid].encode("utf-8")
    if tid in ID_TO_SPECIAL:
        return b"" if skip_special else ID_TO_SPECIAL[tid].encode("utf-8")
    _, inv, _ = load_vocab()
    s = inv.get(tid, "")
    return bytes(BYTE_DEC[c] for c in s if c in BYTE_DEC)


def decode(ids, skip_special=True, keep_tool_tags=False):
    raw = bytearray()
    for i in ids:
        raw.extend(token_to_bytes(i, skip_special=skip_special, keep_tool_tags=keep_tool_tags))
    return raw.decode("utf-8", "replace")


class IncrementalDecoder:
    """Buffers incomplete trailing UTF-8 multi-byte sequences so CJK streaming never emits U+FFFD."""

    def __init__(self, skip_special=True, keep_tool_tags=False):
        self.skip_special = skip_special
        self.keep_tool_tags = keep_tool_tags
        self.buf = bytearray()

    def push(self, tid):
        self.buf.extend(token_to_bytes(tid, skip_special=self.skip_special, keep_tool_tags=self.keep_tool_tags))
        if not self.buf:
            return ""
        # Check up to last 3 bytes for an incomplete UTF-8 lead sequence
        try:
            text = self.buf.decode("utf-8")
            self.buf.clear()
            return text
        except UnicodeDecodeError as e:
            if e.reason == "unexpected end of data" and len(self.buf) - e.start <= 3:
                valid = bytes(self.buf[: e.start]).decode("utf-8", "replace")
                self.buf = self.buf[e.start :]
                return valid
            # Truly invalid byte: flush with replacement
            text = self.buf.decode("utf-8", "replace")
            self.buf.clear()
            return text

    def flush(self):
        if not self.buf:
            return ""
        text = self.buf.decode("utf-8", "replace")
        self.buf.clear()
        return text


@functools.lru_cache(maxsize=1)
def get_chat_template(tok_dir=DEFAULT_TOK_DIR):
    try:
        import frontend as fe
        jpath = os.path.join(tok_dir, "chat_template.jinja")
        if os.path.isfile(jpath):
            return fe.ChatTemplate(jpath)
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=1)
def get_literal_tags():
    try:
        import frontend as fe
        return fe.literal_tags(list(SPECIAL_TOKENS.keys()))
    except Exception:
        return None


def render_and_encode(messages, tools=None, add_generation_prompt=True, **kwargs):
    """Renders official `chat_template.jinja` with literal-tag protection and encodes to token IDs."""
    tmpl = get_chat_template()
    ltags = get_literal_tags()
    if tmpl is not None and ltags is not None:
        import frontend as fe
        marked_msgs, marked_tools, changed = fe.mark_think_literals(messages, tools, ltags)
        prompt = tmpl.render(marked_msgs, tools=marked_tools, add_generation_prompt=add_generation_prompt, **kwargs)
        plain = ()
        if changed:
            prompt, plain = fe.unmark_think_literals(prompt, ltags)
        return encode_with_plain_spans(prompt, plain)
    prompt = format_chat(messages, tools=tools, enable_thinking=kwargs.get("enable_thinking", True))
    return encode(prompt, allow_special=True)
def _render_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def format_chat(messages, tools=None, enable_thinking=True):
    """Formats OpenAI messages into Qwen3.8-Flash-Next ChatML prompt matching chat_template.jinja."""
    out = []
    sys_text = ""
    start_idx = 0
    if messages and messages[0].get("role") == "system":
        sys_text = _render_content(messages[0].get("content")).strip()
        start_idx = 1

    if tools:
        out.append("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for tool in tools:
            out.append("\n" + json.dumps(tool, ensure_ascii=False))
        out.append(
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        if sys_text:
            out.append("\n\n" + sys_text)
        out.append("<|im_end|>\n")
    elif sys_text:
        out.append(f"<|im_start|>system\n{sys_text}<|im_end|>\n")

    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            c = _render_content(messages[idx].get("content")).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                last_user_idx = idx
                break

    for idx in range(start_idx, len(messages)):
        m = messages[idx]
        role = m.get("role", "user")
        content = _render_content(m.get("content")).strip()
        if role == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            tcalls = m.get("tool_calls") or []
            tc_strs = []
            for tc in tcalls:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict) and fn.get("name"):
                    fname = fn["name"]
                    fargs = fn.get("arguments", {})
                    if isinstance(fargs, str):
                        try:
                            fargs = json.loads(fargs)
                        except Exception:
                            fargs = {" raw": fargs}
                    lines = [f"<tool_call>\n<function={fname}>"]
                    if isinstance(fargs, dict):
                        for pk, pv in fargs.items():
                            pval = pv if isinstance(pv, str) else json.dumps(pv, ensure_ascii=False)
                            lines.append(f"<parameter={pk}>\n{pval}\n</parameter>")
                    lines.append("</function>\n</tool_call>")
                    tc_strs.append("\n".join(lines))
            full_body = content
            if tc_strs:
                full_body = (full_body + "\n" + "\n".join(tc_strs)).strip() if full_body else "\n".join(tc_strs)
            rc = (m.get("reasoning_content") or "").strip()
            if idx > last_user_idx and rc:
                out.append(f"<|im_start|>assistant\n<think>\n{rc}\n</think>\n\n{full_body}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>assistant\n{full_body}<|im_end|>\n")
        elif role == "tool":
            out.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")

    if enable_thinking:
        out.append("<|im_start|>assistant\n<think>\n")
    else:
        out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)
