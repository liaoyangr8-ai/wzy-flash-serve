"use strict";

const $ = (id) => document.getElementById(id);
const setTxt = (id, val) => {
  const el = $(id);
  const s = String(val);
  if (el && el.textContent !== s) el.textContent = s;
};
const fmtInt = (n) => (typeof n === "number" && isFinite(n) ? Math.round(n).toLocaleString("en-US") : "—");
const fmtNum = (n, d = 1) => (typeof n === "number" && isFinite(n) ? n.toFixed(d) : "—");

let activeTab = "monitor";
let cfgInitialized = false;
let cfgDirty = false;
let selectedReqId = null;
let selectedDetail = null;
let activeInspPane = "output";
let chatHistory = [];
let actionBusy = false;
let lastMetrics = null;

function showToast(msg, ms = 3200) {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showToast._tid);
  showToast._tid = setTimeout(() => {
    el.hidden = true;
  }, ms);
}

async function apiCall(path, body) {
  const opts = { cache: "no-store", headers: {} };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  const data = await resp.json();
  if (!resp.ok || data.ok === false) {
    throw new Error((data.error && (data.error.message || data.error)) || `HTTP ${resp.status}`);
  }
  return data;
}

// ---------------- Tabs & Theme ----------------
document.querySelectorAll(".st-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll(".st-tab").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    ["monitor", "launcher", "chat", "about"].forEach((t) => {
      const v = $(`view-${t}`);
      if (v) v.hidden = t !== activeTab;
    });
    if (activeTab === "launcher") refreshLogs();
  });
});

$("hdr-port-btn").addEventListener("click", () => {
  $("tab-launcher").click();
  setTimeout(() => $("cfg-port").focus(), 60);
});

$("hdr-theme-btn").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("wzy.theme", next);
  } catch (_) {}
});

// Track form edits in Launcher so 1Hz poll doesn't overwrite user typing
["cfg-port", "cfg-mirrors", "cfg-split", "cfg-ctx", "cfg-adapt", "cfg-threads", "cfg-spec", "cfg-spec-minp", "cfg-temp", "cfg-autoload"].forEach((id) => {
  const el = $(id);
  if (el) el.addEventListener("input", () => { cfgDirty = true; });
});

function collectConfigForm() {
  return {
    port: parseInt($("cfg-port").value || "11438", 10),
    mirror_ports: $("cfg-mirrors").value.trim(),
    split: parseInt($("cfg-split").value || "24", 10),
    ctx_max: parseInt($("cfg-ctx").value || "262144", 10),
    adapt_slots: parseInt($("cfg-adapt").value || "2048", 10),
    cpu_threads: parseInt($("cfg-threads").value || "14", 10),
    spec: parseInt($("cfg-spec").value || "6", 10),
    spec_min_p: parseFloat($("cfg-spec-minp").value || "0.72"),
    temperature: parseFloat($("cfg-temp").value || "0.3"),
    auto_load: $("cfg-autoload").value === "true",
  };
}

function populateConfigForm(cfg) {
  if (!cfg || (cfgInitialized && cfgDirty)) return;
  $("cfg-port").value = cfg.port || 11438;
  $("cfg-mirrors").value = (cfg.mirror_ports || []).join(", ");
  $("cfg-split").value = cfg.split || 24;
  $("cfg-ctx").value = String(cfg.ctx_max || 262144);
  $("cfg-adapt").value = cfg.adapt_slots !== undefined ? cfg.adapt_slots : 2048;
  $("cfg-threads").value = cfg.cpu_threads || 14;
  $("cfg-spec").value = cfg.spec !== undefined ? cfg.spec : 6;
  $("cfg-spec-minp").value = cfg.spec_min_p !== undefined ? cfg.spec_min_p : 0.72;
  $("cfg-temp").value = cfg.temperature !== undefined ? cfg.temperature : 0.3;
  $("cfg-autoload").value = String(Boolean(cfg.auto_load));
  cfgInitialized = true;
}

// ---------------- Launcher Controls ----------------
async function doEngineAction(action) {
  if (actionBusy) return;
  actionBusy = true;
  try {
    if (action === "start") {
      const body = cfgInitialized ? collectConfigForm() : {};
      showToast("正在后台拉起双卡 CUDA 引擎 (~75s)…");
      await apiCall("/api/launcher/start", body);
      cfgDirty = false;
    } else if (action === "stop") {
      showToast("正在关闭引擎并释放 44 GiB 双卡显存…");
      await apiCall("/api/launcher/stop", {});
      showToast("引擎已关闭，双卡显存已全部释放！");
    } else if (action === "restart") {
      const body = cfgInitialized ? collectConfigForm() : {};
      showToast("正在应用配置并重启双卡引擎…");
      await apiCall("/api/launcher/restart", body);
      cfgDirty = false;
    }
    await pollMetrics();
    if (activeTab === "launcher") await refreshLogs();
  } catch (e) {
    showToast("操作失败: " + e.message, 4500);
  } finally {
    actionBusy = false;
  }
}

$("hdr-start-btn").addEventListener("click", () => doEngineAction("start"));
$("hdr-stop-btn").addEventListener("click", () => doEngineAction("stop"));
$("hdr-restart-btn").addEventListener("click", () => doEngineAction("restart"));
$("lch-btn-start").addEventListener("click", () => doEngineAction("start"));
$("lch-btn-stop").addEventListener("click", () => doEngineAction("stop"));
$("lch-btn-restart").addEventListener("click", () => doEngineAction("restart"));

$("lch-btn-save-only").addEventListener("click", async () => {
  try {
    const body = collectConfigForm();
    const res = await apiCall("/api/launcher/config", body);
    cfgDirty = false;
    showToast(res.need_engine_restart ? "采样参数已实时生效；切分层/槽位修改将在下次重启引擎后生效" : "配置已保存并实时生效！");
    await pollMetrics();
  } catch (e) {
    showToast("保存失败: " + e.message, 4500);
  }
});

$("btn-apply-port").addEventListener("click", async () => {
  try {
    const newPort = parseInt($("cfg-port").value || "11438", 10);
    const mirrors = $("cfg-mirrors").value.trim();
    const curPort = parseInt(window.location.port || "80", 10);
    const launcherPort = (lastMetrics && lastMetrics.config && lastMetrics.config.launcher_port) || 11439;
    const res = await apiCall("/api/launcher/port", { port: newPort, mirror_ports: mirrors });
    cfgDirty = false;
    showToast(`服务端口已成功热切换至 :${res.port} (监听列表: ${res.active_ports.map((p) => ":" + p).join(", ")})`);
    if (curPort !== launcherPort && curPort !== res.port && res.active_ports.includes(res.port)) {
      setTimeout(() => {
        window.location.href = `${window.location.protocol}//${window.location.hostname}:${res.port}/`;
      }, 900);
    } else {
      await pollMetrics();
    }
  } catch (e) {
    showToast("端口切换失败: " + e.message, 4500);
  }
});

$("btn-reset-kv").addEventListener("click", async () => {
  try {
    await apiCall("/reset", {});
    showToast("已清空引擎 KV 前缀缓存");
  } catch (e) {
    showToast("清空失败: " + e.message);
  }
});

async function refreshLogs() {
  try {
    const d = await apiCall("/api/logs");
    const box = $("lch-log-box");
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
    box.textContent = (d.logs && d.logs.length) ? d.logs.join("\n") : "暂无引擎日志。";
    if (nearBottom) box.scrollTop = box.scrollHeight;
  } catch (_) {}
}
$("btn-refresh-logs").addEventListener("click", refreshLogs);

// ---------------- Oscilloscope SVG Renderer ----------------
function renderOscilloscope(series) {
  if (!Array.isArray(series) || !series.length) return;
  const W = 720, H = 175, topPad = 18;
  const maxPts = 120;
  const pts = series.slice(-maxPts);
  let peakDec = 80;
  for (const s of pts) {
    if ((s.decode_tps || 0) > peakDec) peakDec = s.decode_tps;
  }
  const yMax = Math.ceil((peakDec * 1.12) / 10) * 10;
  setTxt("osc-ymax", `${yMax} tok/s`);
  setTxt("osc-ymid", `${Math.round(yMax / 2)} tok/s`);

  const n = pts.length;
  const stepX = W / Math.max(1, maxPts - 1);
  const startX = W - (n - 1) * stepX;

  let lineD = "";
  let areaD = "";
  let pfBarsHtml = "";

  for (let i = 0; i < n; i++) {
    const x = startX + i * stepX;
    const dec = Math.max(0, pts[i].decode_tps || 0);
    const y = H - (dec / yMax) * (H - topPad);
    lineD += (i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`);
    if (i === 0) areaD = `M${x.toFixed(1)},${H} L${x.toFixed(1)},${y.toFixed(1)}`;
    else areaD += ` L${x.toFixed(1)},${y.toFixed(1)}`;
    if (i === n - 1) areaD += ` L${x.toFixed(1)},${H} Z`;

    const pf = pts[i].prefill_tps || 0;
    if (pf > 0) {
      const pfH = Math.min(H - topPad, Math.max(8, (pf / 3200.0) * (H - topPad)));
      const pfY = H - pfH;
      pfBarsHtml += `<rect x="${(x - 2.2).toFixed(1)}" y="${pfY.toFixed(1)}" width="4.4" height="${pfH.toFixed(1)}" rx="1.5" fill="var(--st-info)" opacity="0.55"><title>Prefill: ${pf} tok/s</title></rect>`;
    }
  }
  $("osc-dec-line").setAttribute("d", lineD);
  $("osc-dec-area").setAttribute("d", areaD);
  $("osc-pf-bars").innerHTML = pfBarsHtml;
}

// ---------------- Request Detail Inspector ----------------
function updateInspectorPane() {
  if (!selectedDetail) return;
  let txt = "";
  if (activeInspPane === "output") {
    txt = selectedDetail.output || (selectedDetail.reasoning ? `[仅包含 <think> 思考阶段输出]\n\n${selectedDetail.reasoning}` : "(无输出文本或正在流式生成中)");
  } else if (activeInspPane === "reasoning") {
    txt = selectedDetail.reasoning || "(该请求未产生单独的 <think> 思维链内容)";
  } else if (activeInspPane === "input") {
    txt = selectedDetail.input || "(无输入预览)";
  } else {
    const clone = Object.assign({}, selectedDetail);
    delete clone.input;
    delete clone.output;
    delete clone.reasoning;
    delete clone.response;
    txt = JSON.stringify(clone, null, 2);
  }
  setTxt("insp-content", txt);
}

document.querySelectorAll("[data-ipane]").forEach((btn) => {
  btn.addEventListener("click", () => {
    activeInspPane = btn.dataset.ipane;
    document.querySelectorAll("[data-ipane]").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    updateInspectorPane();
  });
});

$("insp-close").addEventListener("click", () => {
  selectedReqId = null;
  selectedDetail = null;
  $("req-inspector").hidden = true;
});

$("insp-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("insp-content").textContent);
    showToast("已复制到剪贴板");
  } catch (_) {}
});

async function openRequestDetail(reqId) {
  selectedReqId = reqId;
  try {
    selectedDetail = await apiCall(`/api/requests?id=${encodeURIComponent(reqId)}`);
    $("req-inspector").hidden = false;
    setTxt("insp-title", `${selectedDetail.id} · ${selectedDetail.path || "/v1/chat/completions"} · Decode ${fmtNum(selectedDetail.decode_tps, 2)} tok/s · Prefill ${fmtNum(selectedDetail.prefill_tps, 1)} tok/s`);
    updateInspectorPane();
  } catch (e) {
    showToast("无法读取请求详情: " + e.message);
  }
}

// ---------------- Requests Table Renderer ----------------
function renderRequestsTable(live, reqs, totals) {
  const tbody = $("req-tbody");
  const rows = [];

  if (live && live.phase) {
    const isPf = live.phase === "prefill";
    const stBadge = isPf ? "预填充 Reading" : "生成中 Generating";
    const pfTps = live.prompt_tps > 0 ? fmtNum(live.prompt_tps, 1) : (isPf ? "计算中…" : "—");
    const decTps = live.inst_tps > 0 ? `${fmtNum(live.inst_tps, 2)} (实时)` : (isPf ? "—" : "启动中…");
    rows.push(`
      <tr class="live-row">
        <td class="mono">${new Date((live.started || Date.now() / 1000) * 1000).toLocaleTimeString()}</td>
        <td><strong style="color:var(--st-warn-text)">● ${stBadge}</strong></td>
        <td class="num">${fmtInt(live.prompt_tokens)}</td>
        <td class="num">${fmtInt(live.reused || 0)}</td>
        <td class="num">${fmtInt(live.read || (live.prompt_tokens - (live.reused || 0)))}</td>
        <td class="num"><strong>${fmtInt(live.generated || 0)}</strong></td>
        <td class="num">${pfTps}</td>
        <td class="num"><strong>${decTps}</strong></td>
        <td class="num">—</td>
        <td class="num">—</td>
        <td class="num">${fmtNum(live.elapsed_s || 0, 1)}s</td>
      </tr>
    `);
  }

  for (const r of (reqs || [])) {
    const pTok = r.prompt_tokens || 0;
    const reused = r.reused || 0;
    const readTok = r.read !== undefined ? r.read : Math.max(1, pTok - reused);
    const pMs = r.prompt_ms || 0;
    const pfTps = r.prefill_tps !== undefined ? r.prefill_tps : (pMs > 0 ? (1000.0 * readTok / pMs) : 0);
    const mtpAcc = r.draft_n > 0 ? `${((100.0 * (r.draft_accepted || 0)) / r.draft_n).toFixed(1)}%` : "—";
    const vhit = r.vram_hit !== undefined ? `${(r.vram_hit * 100).toFixed(1)}%` : "—";
    const durS = r.wallclock_s !== undefined ? r.wallclock_s : (((r.prompt_ms || 0) + (r.decode_ms || 0)) / 1000.0);
    const rid = r.id || "";
    const selCls = rid && rid === selectedReqId ? "selected" : "";
    rows.push(`
      <tr class="${selCls}" data-rid="${rid}">
        <td class="mono">${r.time ? new Date(r.time * 1000).toLocaleTimeString() : "—"}</td>
        <td><span style="color:var(--st-accent-text)">✓ 完成 (${r.finish || "stop"})</span></td>
        <td class="num">${fmtInt(pTok)}</td>
        <td class="num">${fmtInt(reused)}${r.restored_park ? ' <span style="color:var(--st-info-text);font-size:11px">⚡停靠恢复</span>' : ""}</td>
        <td class="num">${fmtInt(readTok)}</td>
        <td class="num">${fmtInt(r.generated || 0)}</td>
        <td class="num" style="color:var(--st-info-text)">${fmtNum(pfTps, 1)}</td>
        <td class="num" style="color:var(--st-accent-text);font-weight:700">${fmtNum(r.decode_tps || 0, 2)}</td>
        <td class="num">${mtpAcc}</td>
        <td class="num">${vhit}${r.promoted ? ` <span class="muted">(+${r.promoted})</span>` : ""}</td>
        <td class="num">${fmtNum(durS, 2)}s</td>
      </tr>
    `);
  }

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="11" class="muted" style="text-align:center;padding:24px">暂无请求记录，API 调用将自动实时显示在此处</td></tr>`;
  } else {
    tbody.innerHTML = rows.join("");
    tbody.querySelectorAll("tr[data-rid]").forEach((tr) => {
      tr.addEventListener("click", () => {
        if (tr.dataset.rid) openRequestDetail(tr.dataset.rid);
      });
    });
  }

  if (totals) {
    setTxt(
      "req-summary",
      `累计请求: ${fmtInt(totals.requests)} · 累计输入: ${fmtInt(totals.prompt_tokens)} tok (缓存复用 ${fmtInt(totals.reused_tokens)} tok) · 累计输出: ${fmtInt(totals.output_tokens)} tok · 会话平均 Prefill: ${fmtNum(totals.prefill_tok_s_mean, 1)} tok/s · 会话平均 Decode: ${fmtNum(totals.decode_tok_s_mean, 2)} tok/s`
    );
  }
}

// ---------------- Main 1Hz Poll ----------------
async function pollMetrics() {
  try {
    const m = await apiCall("/metrics");
    lastMetrics = m;
    const eng = m.engine || {};
    const est = m.engine_state || (eng.vram_slots ? "ready" : "stopped");
    const live = m.live || {};
    const boot = m.boot || {};
    const cfg = m.config || {};
    const reqs = m.requests || [];
    const lastReq = reqs[0] || null;
    const lastPf = m.last_prefill || {};
    const tot = m.totals || {};

    populateConfigForm(cfg);

    // Header port & buttons
    const curPort = cfg.port || 11438;
    setTxt("hdr-port-btn", `API 端口 :${curPort}`);
    setTxt("lch-active-ports", `当前监听端口: ${(cfg.active_ports || [curPort]).map((p) => ":" + p).join(", ")} (固定控制台 :${cfg.launcher_port || 11439})`);
    setTxt("lch-state-badge", `引擎状态: ${est.toUpperCase()}`);
    setTxt("lch-boot-stage", boot.stage || est);
    setTxt("lch-boot-pct", `${boot.pct || 0}%`);
    $("lch-boot-bar").style.width = `${boot.pct || 0}%`;

    const isStopped = est === "stopped" || est === "error";
    $("hdr-start-btn").disabled = actionBusy || !isStopped;
    $("hdr-stop-btn").disabled = actionBusy || isStopped;
    $("lch-btn-start").disabled = actionBusy || !isStopped;
    $("lch-btn-stop").disabled = actionBusy || isStopped;

    // Determine high-level state badge
    let uiState = "idle";
    if (est === "stopped") uiState = "stopped";
    else if (est === "error") uiState = "error";
    else if (est === "booting") uiState = "booting";
    else if (m.busy && live.phase === "prefill") uiState = "reading";
    else if (m.busy) uiState = "generating";

    document.querySelectorAll("#state-badges .st-badge").forEach((b) => {
      b.classList.toggle("active", b.dataset.s === uiState);
    });
    $("hdr-pill").dataset.state = uiState;

    if (uiState === "stopped") {
      setTxt("hdr-pill-text", "已停止 (0 显存)");
      setTxt("state-label", "引擎已关闭 · 44 GiB 双卡显存已释放");
      setTxt("state-detail", "点击右上角「启动引擎」即可重新加载");
      $("state-bar").style.width = "0%";
    } else if (uiState === "booting") {
      setTxt("hdr-pill-text", `启动中 ${boot.pct || 0}%`);
      setTxt("state-label", boot.stage || "正在启动双卡引擎…");
      setTxt("state-detail", `已耗时 ${fmtNum(boot.elapsed_s || 0, 1)}s`);
      $("state-bar").style.width = `${boot.pct || 10}%`;
    } else if (uiState === "reading") {
      setTxt("hdr-pill-text", `预填充 ${fmtInt(live.prompt_tokens)} tok`);
      setTxt("state-label", `正在读取并预填充 Prompt (${fmtInt(live.prompt_tokens)} tokens)…`);
      setTxt("state-detail", `已耗时 ${fmtNum(live.elapsed_s || 0, 1)}s`);
      $("state-bar").style.width = "40%";
    } else if (uiState === "generating") {
      const liveTps = live.inst_tps || live.avg_tps || 0;
      setTxt("hdr-pill-text", `生成中 ${fmtNum(liveTps, 1)} tok/s`);
      setTxt(
        "state-label",
        `正在流式生成第 ${fmtInt(live.generated)} 个 Token · 实时速度 ${fmtNum(liveTps, 2)} tok/s (本轮均值 ${fmtNum(live.avg_tps || 0, 2)} tok/s)`
      );
      setTxt(
        "state-detail",
        `Prompt ${fmtInt(live.prompt_tokens)} (缓存复用 ${fmtInt(live.reused || 0)}) · Prefill ${fmtNum(live.prompt_tps || 0, 1)} tok/s`
      );
      const pct = Math.min(99, Math.max(6, ((live.generated || 1) / Math.max(1, live.max_tokens || 2048)) * 100));
      $("state-bar").style.width = `${pct.toFixed(1)}%`;
    } else {
      setTxt("hdr-pill-text", `就绪 · :${curPort}`);
      setTxt("state-label", `双卡引擎空闲就绪 · 等待 API 请求 (http://${window.location.hostname}:${curPort}/v1)`);
      setTxt("state-detail", `${eng.vram_slots || 13224} 显存专家槽 · 最大上下文 ${fmtInt(eng.ctx_max || cfg.ctx_max || 262144)}`);
      $("state-bar").style.width = "100%";
    }

    // Card 1: Real-time Decode Speed
    if (uiState === "generating") {
      setTxt("m-dec-tag", "实时 2s 窗口");
      setTxt("m-dec-tps", fmtNum(live.inst_tps || live.avg_tps || 0, 2));
      setTxt("m-dec-sub", `本轮均值 ${fmtNum(live.avg_tps || 0, 2)} tok/s · 已输出 ${fmtInt(live.generated)} tok`);
      setTxt("osc-live-note", `正在输出: ${fmtNum(live.inst_tps || 0, 2)} tok/s`);
    } else if (lastReq) {
      setTxt("m-dec-tag", "最近一轮");
      setTxt("m-dec-tps", fmtNum(lastReq.decode_tps || 0, 2));
      setTxt("m-dec-sub", `会话均值 ${fmtNum(tot.decode_tok_s_mean || 0, 2)} tok/s · 累计 ${fmtInt(tot.output_tokens || 0)} tok`);
      setTxt("osc-live-note", `最近一轮: ${fmtNum(lastReq.decode_tps || 0, 2)} tok/s`);
    } else {
      setTxt("m-dec-tps", "—");
      setTxt("m-dec-sub", `会话均值 ${fmtNum(tot.decode_tok_s_mean || 0, 2)} tok/s`);
    }

    // Card 2: Prefill Speed
    const pfShow = (m.busy && live.prompt_tps > 0) ? live : (lastPf.prompt_tps ? lastPf : lastReq);
    if (pfShow && (pfShow.prompt_tps || pfShow.prefill_tps)) {
      const pTps = pfShow.prompt_tps || pfShow.prefill_tps || 0;
      const pTok = pfShow.prompt_tokens || 0;
      const rTok = pfShow.reused || 0;
      const pMs = pfShow.prompt_ms || 0;
      setTxt("m-pf-tps", fmtNum(pTps, 1));
      setTxt("m-pf-sub", `Prompt ${fmtInt(pTok)} (复用 ${fmtInt(rTok)}) · ${fmtNum(pMs, 1)} ms`);
    } else {
      setTxt("m-pf-tps", tot.prefill_tok_s_mean ? fmtNum(tot.prefill_tok_s_mean, 1) : "—");
      setTxt("m-pf-sub", "等待首条请求预填充…");
    }

    // Card 3 & 4: MTP Acceptance & VRAM Hit
    if (lastReq) {
      const dN = lastReq.draft_n || 0;
      const dA = lastReq.draft_accepted || 0;
      setTxt("m-mtp-acc", dN > 0 ? ((100.0 * dA) / dN).toFixed(1) : "0.0");
      setTxt("m-mtp-sub", `命中 ${fmtInt(dA)} / 提议 ${fmtInt(dN)} 草稿 token`);
      setTxt("m-vram-hit", lastReq.vram_hit !== undefined ? (lastReq.vram_hit * 100).toFixed(2) : "—");
      setTxt("m-vram-sub", `${fmtInt(eng.vram_slots || 13224)} 常驻槽 · 本轮晋升 +${fmtInt(lastReq.promoted || 0)}`);
    } else {
      setTxt("m-vram-sub", `${fmtInt(eng.vram_slots || 0)} / 24,576 常驻专家槽`);
    }

    // Card 5, 6, 7: Dual GPUs
    const gpus = m.gpus || [];
    if (gpus.length >= 2) {
      const g0 = gpus[0], g1 = gpus[1];
      setTxt("m-gpu-util", `${g0.util}% / ${g1.util}%`);
      if (g0.sm_mhz) setTxt("m-clock-tag", `SM ${g0.sm_mhz} MHz`);
      const sp = eng.split || cfg.split || 24;
      setTxt("m-gpu-split", `PP2 对称切分 L0..${sp - 1} | L${sp}..47`);

      const totGib = (g0.mem_mib + g1.mem_mib) / 1024.0;
      setTxt("m-vram-gib", totGib.toFixed(1));
      setTxt("m-vram-cards", `G0: ${fmtInt(g0.mem_mib)} MiB · G1: ${fmtInt(g1.mem_mib)} MiB`);

      const maxT = Math.max(g0.temp, g1.temp);
      const sumP = (g0.power_w + g1.power_w).toFixed(0);
      setTxt("m-temp-pwr", `${maxT}°C / ${sumP}W`);
      setTxt("m-temp-sub", `G0: ${g0.temp}°C ${g0.power_w}W · G1: ${g1.temp}°C ${g1.power_w}W`);

      // Individual GPU strips
      setTxt("g0-vram-txt", `${fmtInt(g0.mem_mib)} / ${fmtInt(g0.mem_total_mib)} MiB`);
      $("g0-vram-bar").style.width = `${Math.min(100, (100 * g0.mem_mib) / g0.mem_total_mib).toFixed(1)}%`;
      setTxt("g0-stat-txt", `${g0.util}% · ${g0.temp}°C · ${g0.power_w}W`);
      $("g0-util-bar").style.width = `${g0.util}%`;

      setTxt("g1-vram-txt", `${fmtInt(g1.mem_mib)} / ${fmtInt(g1.mem_total_mib)} MiB`);
      $("g1-vram-bar").style.width = `${Math.min(100, (100 * g1.mem_mib) / g1.mem_total_mib).toFixed(1)}%`;
      setTxt("g1-stat-txt", `${g1.util}% · ${g1.temp}°C · ${g1.power_w}W`);
      $("g1-util-bar").style.width = `${g1.util}%`;
    }

    // Card 8 & Context Gauge
    const sys = m.sys || {};
    setTxt("m-ram-gib", sys.ram_used_gib !== undefined ? sys.ram_used_gib.toFixed(1) : "—");
    setTxt("m-port-sub", `API :${curPort} · 控制台 :${cfg.launcher_port || 11439}`);
    setTxt("bar-ram-txt", `${sys.ram_used_gib || 0} / ${sys.ram_total_gib || 94.3} GiB`);
    $("bar-ram-fill").style.width = `${sys.ram_pct || 0}%`;

    const ctxMax = eng.ctx_max || cfg.ctx_max || 262144;
    const sslots = m.session_slots || {};
    setTxt("ctx-cap-label", sslots.parked_tokens > 0 ? `停靠保护 ${fmtInt(sslots.parked_tokens)} tok · Max ${fmtInt(ctxMax)}` : `Max ${fmtInt(ctxMax)}`);
    const activeCtx = m.busy
      ? (live.prompt_tokens || 0) + (live.generated || 0)
      : lastReq
      ? (lastReq.prompt_tokens || 0) + (lastReq.generated || 0)
      : 0;
    const ctxRatio = Math.min(1, activeCtx / Math.max(1, ctxMax));
    $("ctx-fill").setAttribute("stroke-dasharray", `${(ctxRatio * 235.6).toFixed(1)} 314.2`);
    setTxt("ctx-pct", `${(ctxRatio * 100).toFixed(1)}%`);
    setTxt("ctx-sub", `${fmtInt(activeCtx)} tok`);

    const vSlots = eng.vram_slots || 0;
    setTxt("bar-slots-txt", `${fmtInt(vSlots)} / 24,576 (${((100 * vSlots) / 24576).toFixed(1)}%)`);
    $("bar-slots-fill").style.width = `${((100 * vSlots) / 24576).toFixed(1)}%`;

    const kvUsed = Math.min(32768, activeCtx);
    setTxt("bar-kv-txt", `${fmtInt(kvUsed)} / 32,768 (含 4K Sink 锁页)`);
    $("bar-kv-fill").style.width = `${((100 * kvUsed) / 32768).toFixed(1)}%`;

    renderOscilloscope(m.series || []);
    renderRequestsTable(m.busy ? live : null, reqs, tot);

    // Update API & About panels
    const hostIp = window.location.hostname || "192.168.2.100";
    setTxt(
      "api-endpoints-box",
      `# OpenAI Base URL (DeepSeek Harness / Cursor / OpenAI SDK)\n` +
        `http://${hostIp}:${curPort}/v1\n\n` +
        `# Chat Completions Endpoint\n` +
        `POST http://${hostIp}:${curPort}/v1/chat/completions\n\n` +
        `# 模型名称 (Model ID)\n` +
        `${cfg.model_name || "qwen3.8-flash-next-gptq-nvfp4"}\n\n` +
        `# 固定启动器控制台备份入口 (即使切换 API 端口也永久可达)\n` +
        `http://${hostIp}:${cfg.launcher_port || 11439}/\n\n` +
        `# 快速命令行验证\n` +
        `curl http://${hostIp}:${curPort}/v1/chat/completions \\\n` +
        `  -H "Content-Type: application/json" \\\n` +
        `  -d '{"model":"${cfg.model_name || "qwen3.8-flash-next-gptq-nvfp4"}","messages":[{"role":"user","content":"你好"}],"stream":true}'`
    );

    setTxt(
      "about-hw-box",
      `引擎内核 : wzy-pp2-v2-tc-mtp (100% 纯 CUDA/C++ 双卡流水线, 零专家跳过)\n` +
        `模型权重 : Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata\n` +
        `加速显卡 : 2× Unlocked RTX 2080 Ti 22GB (TU102 sm_75 Tensor Core + NVLink)\n` +
        `层切分比 : GPU0 Layers [0..${(eng.split || 24) - 1}] + PLE | GPU1 Layers [${eng.split || 24}..47] + Head + MTP\n` +
        `驻留专家 : ${fmtInt(vSlots)} / 24,576 槽位 (含双卡 ${fmtInt((cfg.adapt_slots || 2048) * 2)} 动态自适应热晋升槽)\n` +
        `KV 架构  : 256K 稀疏 QSA + 32,768 GDDR6 显存环 (前 4,096 Attention-Sink 永驻锁页)\n` +
        `投机解码 : 原生 Q2_0 MTP 草稿头 + 114,452 中英混合草稿词表 (K=${cfg.spec || 6}, min_p=${cfg.spec_min_p || 0.72})`
    );
  } catch (e) {
    $("hdr-pill").dataset.state = "error";
    setTxt("hdr-pill-text", "连接断开");
  }
}

// ---------------- Built-in Streaming Chat ----------------
function appendChatBox(role, text) {
  const empty = $("chat-empty");
  if (empty) empty.hidden = true;
  const wrap = $("chat-messages");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.innerHTML = `<div class="chat-msg__role">${role === "user" ? "User" : "Qwen3.8-Flash-Next NVFP4"}</div><div class="holder"></div><div class="txt" style="white-space:pre-wrap"></div><div class="muted small mono meta" style="margin-top:8px"></div>`;
  div.querySelector(".txt").textContent = text || "";
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return div;
}

async function sendChatMessage(overrideText) {
  const inp = $("chat-input");
  const txt = (overrideText !== undefined ? overrideText : inp.value).trim();
  if (!txt) return;
  if (overrideText === undefined) inp.value = "";
  $("chat-send").disabled = true;

  chatHistory.push({ role: "user", content: txt });
  appendChatBox("user", txt);
  const box = appendChatBox("assistant", "");
  const holder = box.querySelector(".holder");
  const txtEl = box.querySelector(".txt");
  const metaEl = box.querySelector(".meta");

  let thinkEl = null, rc = "", ans = "";
  const t0 = performance.now();
  let tokCount = 0;

  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "qwen3.8-flash-next-gptq-nvfp4",
        messages: chatHistory,
        stream: true,
        enable_thinking: $("chat-think").checked,
        temperature: parseFloat($("chat-temp").value || "0.3"),
        max_tokens: parseInt($("chat-max").value || "512", 10),
      }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error((err.error && err.error.message) || `HTTP ${resp.status}`);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const blk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const ln of blk.split("\n")) {
          if (!ln.startsWith("data: ")) continue;
          const payload = ln.slice(6).trim();
          if (payload === "[DONE]") break;
          const j = JSON.parse(payload);
          if (j.error) throw new Error(j.error.message || "生成异常");
          const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
          if (d.reasoning_content) {
            if (!thinkEl) {
              thinkEl = document.createElement("details");
              thinkEl.className = "chat-think";
              thinkEl.open = true;
              thinkEl.innerHTML = `<summary>思考过程 (&lt;think&gt;)</summary><div class="rc" style="white-space:pre-wrap;margin-top:6px"></div>`;
              holder.appendChild(thinkEl);
            }
            rc += d.reasoning_content;
            tokCount++;
            thinkEl.querySelector(".rc").textContent = rc;
          }
          if (d.content) {
            ans += d.content;
            tokCount++;
            txtEl.textContent = ans;
          }
          const dt = (performance.now() - t0) / 1000.0;
          if (!j.timings && dt > 0.2) {
            metaEl.textContent = `流式输出中… 已接收 ${tokCount} 段 · 约 ${(tokCount / dt).toFixed(1)} chunks/s`;
          }
          if (j.timings) {
            const tm = j.timings;
            metaEl.textContent = `Prefill: ${tm.prompt_n} tok (${tm.cache_n} 复用) @ ${tm.prompt_per_second} tok/s | Decode: ${tm.predicted_n} tok @ ${tm.predicted_per_second} tok/s | MTP 命中 ${tm.draft_accepted}/${tm.draft_n} | VRAM ${(tm.vram_hit * 100).toFixed(1)}%`;
          }
        }
        $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
      }
    }
    chatHistory.push({ role: "assistant", content: ans, reasoning_content: rc });
  } catch (e) {
    txtEl.textContent = "请求失败: " + e.message;
  } finally {
    $("chat-send").disabled = false;
    pollMetrics();
  }
}

$("chat-send").addEventListener("click", () => sendChatMessage());
$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});
$("chat-clear").addEventListener("click", async () => {
  chatHistory = [];
  $("chat-messages").innerHTML = "";
  await apiCall("/reset", {});
  showToast("对话与 KV 缓存已重置");
});
document.querySelectorAll("[data-prompt]").forEach((btn) => {
  btn.addEventListener("click", () => sendChatMessage(btn.dataset.prompt));
});

// Start polling loop
async function loop() {
  await pollMetrics();
  if (activeTab === "launcher") await refreshLogs();
  setTimeout(loop, activeTab === "monitor" || activeTab === "launcher" ? 1000 : 2000);
}
loop();
