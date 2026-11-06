#pragma once
// wzy_setup.cuh -- shared setup/sampling/JSON helpers, extracted verbatim from wzy_engine_v2.cu
#include "wzy_stage_v2.cuh"
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <algorithm>

static bool is_qsa_layer(int l) { return (l % 4) == 3; }

struct GenCfg {
    int max_new = 512;
    float temp = 0.0f;
    float top_p = 0.75f;
    int top_k = 15;
    float min_p = 0.02f;
    float rep_pen = 1.0f;
    float spec_min_p = 0.72f;
    int spec_k = 7;
    bool no_reuse = false;
    std::vector<int> stops = {248046, 248044};
};

static constexpr size_t REP_PEN_WINDOW = 512;
static constexpr float REP_PEN_EPS = 1e-5f;

// Build filtered target distribution over top_k / top_p / min_p; returns (ids, probs) summing to 1.0
static void build_target_dist(const float* logits, int nvocab, const GenCfg& cfg,
                              const std::vector<int>& history,
                              std::vector<int>& out_ids, std::vector<float>& out_probs) {
    out_ids.clear();
    out_probs.clear();
    const bool has_rep = (std::fabs(cfg.rep_pen - 1.0f) >= REP_PEN_EPS) && !history.empty();
    if (cfg.temp <= 1e-5f && !has_rep) {
        int best = 0;
        for (int v = 1; v < nvocab; ++v) if (logits[v] > logits[best]) best = v;
        out_ids.push_back(best);
        out_probs.push_back(1.0f);
        return;
    }

    static std::vector<float> buf;
    const float* src = logits;
    if (has_rep) {
        buf.assign(logits, logits + nvocab);
        const size_t start = history.size() > REP_PEN_WINDOW ? history.size() - REP_PEN_WINDOW : 0;
        static std::vector<uint8_t> seen;
        seen.assign(nvocab, 0);
        for (size_t i = start; i < history.size(); ++i) {
            const int id = history[i];
            if (id >= 0 && id < nvocab && !seen[id]) {
                seen[id] = 1;
                buf[id] = (buf[id] > 0.f) ? (buf[id] / cfg.rep_pen) : (buf[id] * cfg.rep_pen);
            }
        }
        src = buf.data();
    }
    if (cfg.temp <= 1e-5f) {
        int best = 0;
        for (int v = 1; v < nvocab; ++v) if (src[v] > src[best]) best = v;
        out_ids.push_back(best);
        out_probs.push_back(1.0f);
        return;
    }

    const int k = std::min(128, (cfg.top_k > 0 && cfg.top_k < nvocab) ? cfg.top_k : 64);
    float top_v[128];
    int top_i[128];
    int n_top = 0;
    float min_v = -1e30f;
    int min_pos = 0;
    for (int v = 0; v < nvocab; ++v) {
        const float x = src[v];
        if (n_top < k) {
            top_v[n_top] = x;
            top_i[n_top] = v;
            ++n_top;
            if (n_top == k) {
                min_v = top_v[0]; min_pos = 0;
                for (int j = 1; j < k; ++j) if (top_v[j] < min_v) { min_v = top_v[j]; min_pos = j; }
            }
        } else if (x > min_v) {
            top_v[min_pos] = x;
            top_i[min_pos] = v;
            min_v = top_v[0]; min_pos = 0;
            for (int j = 1; j < k; ++j) if (top_v[j] < min_v) { min_v = top_v[j]; min_pos = j; }
        }
    }
    // Insertion sort `n_top <= 128` elements descending in L1 cache!
    for (int i = 1; i < n_top; ++i) {
        const float xv = top_v[i];
        const int xi = top_i[i];
        int j = i - 1;
        while (j >= 0 && (top_v[j] < xv || (top_v[j] == xv && top_i[j] > xi))) {
            top_v[j + 1] = top_v[j];
            top_i[j + 1] = top_i[j];
            --j;
        }
        top_v[j + 1] = xv;
        top_i[j + 1] = xi;
    }

    const float inv_t = 1.0f / cfg.temp;
    const float mx = top_v[0] * inv_t;
    float probs[128];
    float sum = 0.f;
    for (int i = 0; i < n_top; ++i) {
        probs[i] = std::exp(top_v[i] * inv_t - mx);
        sum += probs[i];
    }
    for (int i = 0; i < n_top; ++i) probs[i] /= sum;
    const float p_max = probs[0];
    int keep = n_top;
    float cum = 0.f;
    for (int i = 0; i < n_top; ++i) {
        if (i > 0 && cfg.min_p > 0.f && probs[i] < cfg.min_p * p_max) { keep = i; break; }
        cum += probs[i];
        if (cfg.top_p > 0.f && cfg.top_p < 1.0f && cum >= cfg.top_p) { keep = i + 1; break; }
    }
    float psum = 0.f;
    for (int i = 0; i < keep; ++i) psum += probs[i];
    out_ids.resize(keep);
    out_probs.resize(keep);
    for (int i = 0; i < keep; ++i) {
        out_ids[i] = top_i[i];
        out_probs[i] = probs[i] / psum;
    }
}

static int draw_from_dist(const std::vector<int>& ids, const std::vector<float>& probs, std::mt19937& rng) {
    if (ids.size() == 1) return ids[0];
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(rng);
    for (size_t i = 0; i < ids.size(); ++i) {
        r -= probs[i];
        if (r <= 0.f) return ids[i];
    }
    return ids.back();
}

// Exact lossless speculative verification against deterministic draft token `draft_tok` (q(draft_tok) = 1):
// Accepts `draft_tok` with probability p_target(draft_tok); if rejected, samples from norm(p_target \ {draft_tok})!
static bool verify_or_resample(const float* logits, int nvocab, const GenCfg& cfg,
                               const std::vector<int>& history, int draft_tok,
                               int& out_tok, std::mt19937& rng) {
    static std::vector<int> ids;
    static std::vector<float> probs;
    build_target_dist(logits, nvocab, cfg, history, ids, probs);
    if (ids.size() == 1) {
        out_tok = ids[0];
        return (draft_tok >= 0 && draft_tok == ids[0]);
    }
    static const int typ_rank_v = std::getenv("WZY_TYP_RANK") ? std::atoi(std::getenv("WZY_TYP_RANK")) : 1;
    static const float typ_margin_v = std::getenv("WZY_TYP_MARGIN") ? (float) std::atof(std::getenv("WZY_TYP_MARGIN")) : 0.0f;
    if (typ_rank_v > 1 && draft_tok >= 0) {
        const int rr = std::min(typ_rank_v, (int) ids.size());
        for (int i = 0; i < rr; ++i) {
            if (ids[i] != draft_tok) continue;
            if (i > 0 && typ_margin_v > 0.f) break;   // full-vocab path has no logits here: stay strict
            out_tok = draft_tok;
            return true;
        }
    }
    const int sampled = draw_from_dist(ids, probs, rng);
    if (draft_tok < 0) {
        out_tok = sampled;
        return false;
    }
    out_tok = sampled;
    return (draft_tok >= 0 && draft_tok == sampled);
}

// Zero-PCIe-D2H exact unbiased speculative verification & sampling using GPU-extracted descending Top-16 `(top_v, top_i)`!
static bool verify_or_resample_top16(const float* raw_top_v, const int* raw_top_i, const GenCfg& cfg,
                                     const std::vector<int>& history, size_t gen_start,
                                     int draft_tok, int& out_tok, std::mt19937& rng,
                                     int* slack_left = nullptr) {
    const int k = std::min(16, cfg.top_k > 0 ? cfg.top_k : 16);
    float top_v[16];
    int top_i[16];
    for (int i = 0; i < k; ++i) {
        top_v[i] = raw_top_v[i];
        top_i[i] = raw_top_i[i];
    }

    const size_t hlen = history.size();
    const bool use_rep = (std::fabs(cfg.rep_pen - 1.0f) >= REP_PEN_EPS);
    bool mod = false;
    if (use_rep && hlen > 0) {
        const size_t w0 = hlen > REP_PEN_WINDOW ? (hlen - REP_PEN_WINDOW) : 0;
        for (int i = 0; i < k; ++i) {
            const int tid = top_i[i];
            bool seen = false;
            for (size_t p = w0; p < hlen; ++p) {
                if (history[p] == tid) { seen = true; break; }
            }
            if (seen) {
                top_v[i] = (top_v[i] > 0.f) ? (top_v[i] / cfg.rep_pen) : (top_v[i] * cfg.rep_pen);
                mod = true;
            }
        }
    }
    // Optional heuristic: can alter legitimate repeated text and tool arguments.
    if (std::getenv("WZY_TANDEM_GUARD") && hlen > gen_start + 4) {
        const size_t gen_len = hlen - gen_start;
        const size_t max_L = std::min<size_t>(32, gen_len / 2);
        for (size_t L = 2; L <= max_L; ++L) {
            bool tandem = true;
            for (size_t m = 1; m <= L; ++m) {
                if (history[hlen - m] != history[hlen - L - m]) { tandem = false; break; }
            }
            if (tandem) {
                const int next_loop_tok = history[hlen - L];
                for (int i = 0; i < k; ++i) {
                    if (top_i[i] == next_loop_tok) { top_v[i] -= 6.0f; mod = true; }
                }
            }
        }
    }
    if (mod) {
        for (int i = 1; i < k; ++i) {
            const float v = top_v[i];
            const int id = top_i[i];
            int j = i - 1;
            while (j >= 0 && (top_v[j] < v || (top_v[j] == v && top_i[j] > id))) {
                top_v[j + 1] = top_v[j];
                top_i[j + 1] = top_i[j];
                --j;
            }
            top_v[j + 1] = v;
            top_i[j + 1] = id;
        }
    }
    // WZY_TYP_RANK>1: Medusa/vLLM-style "top-R acceptance" -- accept the drafted token when it lands
    // inside the target's top-R candidates instead of demanding an exact argmax / exact resample hit.
    // rank<=1 keeps the original, bit-exact speculative verification.
    static const int typ_rank = std::getenv("WZY_TYP_RANK") ? std::atoi(std::getenv("WZY_TYP_RANK")) : 1;
    // WZY_TYP_MARGIN: logit-margin guard. A non-argmax draft is only honoured while the target is
    // genuinely undecided about the top candidates (top_v[0] - top_v[i] <= margin). Where the target
    // is confident by a wide margin, only its argmax is accepted (bit-exact behaviour there).
    // UNIT: top_v[] comes straight from row_top16_multi_kernel, i.e. RAW logits -- temperature is NOT
    // divided out here (cfg.temp is only applied later, in the softmax below). So the effective
    // strictness of this knob is (margin / T): at T=0.3 a margin of 1.8 already means a second-place
    // probability ratio of e^-6 ~ 1/400, which effectively disables the whole feature. Calibrate by
    // measurement, not by intuition; 5.0 is the measured knee at T=0.3 (see README section 4).
    static const float typ_margin = std::getenv("WZY_TYP_MARGIN") ? (float) std::atof(std::getenv("WZY_TYP_MARGIN")) : 0.0f;
    // WZY_TYP_MAX_SLACK: per-window budget of sub-argmax acceptances. Once the budget is spent the
    // remaining rows of the same window fall back to strict argmax, so a draft cannot chain several
    // low-probability repetition tokens inside one window and drag the context into a loop.
    const int allow_rank = (slack_left == nullptr || *slack_left > 0) ? typ_rank : 1;
    if (allow_rank > 1 && draft_tok >= 0) {
        const int rr = std::min(allow_rank, k);
        for (int i = 0; i < rr; ++i) {
            if (top_i[i] != draft_tok) continue;
            if (i > 0) {
                if (typ_margin > 0.f && (top_v[0] - top_v[i]) > typ_margin) break;
                if (slack_left != nullptr) --(*slack_left);
            }
            out_tok = draft_tok;
            return true;
        }
    }
    if (cfg.temp <= 1e-5f) {
        out_tok = top_i[0];
        return (draft_tok >= 0 && draft_tok == top_i[0]);
    }
    const float inv_t = 1.0f / cfg.temp;
    const float mx = top_v[0] * inv_t;
    float probs[16];
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(top_v[i] * inv_t - mx);
        sum += probs[i];
    }
    for (int i = 0; i < k; ++i) probs[i] /= sum;
    const float p_max = probs[0];
    int keep = k;
    float cum = 0.f;
    for (int i = 0; i < k; ++i) {
        if (i > 0 && cfg.min_p > 0.f && probs[i] < cfg.min_p * p_max) { keep = i; break; }
        cum += probs[i];
        if (cfg.top_p > 0.f && cfg.top_p < 1.0f && cum >= cfg.top_p) { keep = i + 1; break; }
    }
    float psum = 0.f;
    for (int i = 0; i < keep; ++i) psum += probs[i];
    for (int i = 0; i < keep; ++i) probs[i] /= psum;
    if (keep == 1) {
        out_tok = top_i[0];
        return (draft_tok >= 0 && draft_tok == top_i[0]);
    }
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    float r = u01(rng);
    int sampled = top_i[keep - 1];
    for (int i = 0; i < keep; ++i) {
        r -= probs[i];
        if (r <= 0.f) { sampled = top_i[i]; break; }
    }
    out_tok = sampled;
    return (draft_tok >= 0 && draft_tok == sampled);
}

static void init_stage_weights_and_state(StageGpu& s, int dev, int lb, int le, int ctx_max,
                                         const Gguf& g, const std::vector<float>& h_cos, const std::vector<float>& h_sin) {
    s.dev = dev;
    s.l_begin = lb;
    s.l_end = le;
    s.ctx_max = ctx_max;
    s.has_ple = (lb <= 1 && 1 < le);
    s.has_head = (le == 48);
    CHECK(cudaSetDevice(dev));
    fill_tables_current_device();
    CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&s.abort_stream, cudaStreamNonBlocking));
    cublasCreate(&s.cublas);
    cublasSetStream(s.cublas, s.stream);
    cublasSetMathMode(s.cublas, CUBLAS_DEFAULT_MATH);

    s.L.resize(48);
    s.conv.resize(48);
    s.ssm.resize(48);
    s.n_qsa = 0;
    s.n_gdn = 0;

    for (int l = lb; l < le; ++l) {
        LayerW& w = s.L[l];
        w.qsa = is_qsa_layer(l);
        if (w.qsa) w.qsa_local = s.n_qsa++;
        else w.gdn_local = s.n_gdn++;

        char name[256];
        auto q = [&](const char* suffix) { std::snprintf(name, sizeof name, "blk.%d.%s", l, suffix); return name; };
        for (int h = 0; h < 2; ++h) {
            const char* pre = h == 0 ? "hc_attn_" : "hc_ffn_";
            std::snprintf(name, sizeof name, "blk.%d.%snorm.weight", l, pre);
            w.hc_norm[h] = up_f32(g, name);
            std::snprintf(name, sizeof name, "blk.%d.%sdown.weight", l, pre);
            w.hc_down[h] = up_bf_as_q8(g, name);
            std::snprintf(name, sizeof name, "blk.%d.%sup.weight", l, pre);
            w.hc_up[h] = up_bf_as_q8(g, name);
            std::snprintf(name, sizeof name, "blk.%d.%sinject.weight", l, pre);
            w.hc_inj[h] = up_bf(g, name);
        }
        if (!w.qsa) {
            w.qkv = up_q8(g, q("attn_qkv.weight"));
            w.wgate = up_q8(g, q("attn_gate.weight"));
            w.wout = up_q8(g, q("ssm_out.weight"));
            w.alpha = up_bf(g, q("ssm_alpha.weight"));
            w.beta = up_bf(g, q("ssm_beta.weight"));
            w.ssm_a = up_f32(g, q("ssm_a"));
            w.dtb = up_f32(g, q("ssm_dt.bias"));
            w.conv1d = up_f32(g, q("ssm_conv1d.weight"));
            w.ssm_norm = up_f32(g, q("ssm_norm.weight"));
            s.conv[l] = alloc0((size_t) QKV * 3);
            s.ssm[l] = alloc0((size_t) VH * ST * ST);
        } else {
            w.wq = up_q8(g, q("attn_q.weight"));
            w.wk = up_q8(g, q("attn_k.weight"));
            w.wv = up_q8(g, q("attn_v.weight"));
            w.wo = up_q8(g, q("attn_output.weight"));
            w.q_norm = up_f32(g, q("attn_q_norm.weight"));
            w.k_norm = up_f32(g, q("attn_k_norm.weight"));
            w.idx_q = up_bf(g, q("indexer.q_proj.weight"));
            w.idx_k = up_bf(g, q("indexer.k_proj.weight"));
            w.idx_qn = up_f32(g, q("indexer.q_norm.weight"));
            w.idx_kn = up_f32(g, q("indexer.k_norm.weight"));
        }
        w.g_inp = up_bf(g, q("ffn_gate_inp.weight"));
        w.s_gate_inp = up_bf(g, q("ffn_gate_inp_shexp.weight"));
        w.sh_gate = up_q8(g, q("ffn_gate_shexp.weight"));
        w.sh_up = up_q8(g, q("ffn_up_shexp.weight"));
        w.sh_down = up_q8(g, q("ffn_down_shexp.weight"));
    }

    // GDN verify window snapshots
    s.conv_snaps = alloc0((size_t) s.n_gdn * MAX_W * QKV * 3);
    s.ssm_snaps = alloc0((size_t) s.n_gdn * MAX_W * VH * ST * ST);

    // 256K RoPE tables
    s.cos_tab = alloc(h_cos.size());
    s.sin_tab = alloc(h_sin.size());
    CHECK(cudaMemcpy(s.cos_tab.p, h_cos.data(), h_cos.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s.sin_tab.p, h_sin.data(), h_sin.size() * 4, cudaMemcpyHostToDevice));

    // 256K QSA Sparse Indexer + INT8 Tiered KV Cache
    const int max_b = ctx_max / IDX_BLK + 2;
    s.idx_tail   = alloc0((size_t) s.n_qsa * (IDX_BLK - 1) * IDX_DIM);
    s.idx_dead   = alloc0((size_t) s.n_qsa * IDX_DIM);
    s.idx_pooled = alloc0((size_t) s.n_qsa * MAX_IDX_BLOCKS * IDX_DIM);
    s.idx_scores = alloc0((size_t) MAX_BATCH * max_b);
    s.idx_tail_snaps = alloc0((size_t) s.n_qsa * MAX_W * (IDX_BLK - 1) * IDX_DIM);
    s.idx_dead_snaps = alloc0((size_t) s.n_qsa * MAX_W * IDX_DIM);
    s.idx_pool_snaps = alloc0((size_t) s.n_qsa * MAX_W * 2 * IDX_DIM);
    s.R_handoff = alloc0((size_t) MAX_W * HCD);
    CHECK(cudaMalloc(&s.sel_ids, (size_t) MAX_BATCH * IDX_TOPK * 4));
    CHECK(cudaMalloc(&s.sel_cnt, (size_t) MAX_BATCH * 4));

    CHECK(cudaMalloc(&s.vk_q, (size_t) s.n_qsa * KV_VRAM_RING * KVD));
    CHECK(cudaMalloc(&s.vv_q, (size_t) s.n_qsa * KV_VRAM_RING * KVD));
    CHECK(cudaMalloc(&s.vk_s, (size_t) s.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * 2));
    CHECK(cudaMalloc(&s.vv_s, (size_t) s.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * 2));
    CHECK(cudaMemset(s.vk_q, 0, (size_t) s.n_qsa * KV_VRAM_RING * KVD));
    CHECK(cudaMemset(s.vv_q, 0, (size_t) s.n_qsa * KV_VRAM_RING * KVD));
    CHECK(cudaMemset(s.vk_s, 0, (size_t) s.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * 2));
    CHECK(cudaMemset(s.vv_s, 0, (size_t) s.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * 2));

    if (ctx_max > KV_VRAM_RING) {
        const size_t q_bytes = (size_t) s.n_qsa * ctx_max * KVD;
        const size_t s_bytes = (size_t) s.n_qsa * ctx_max * KV_SC_PER_CELL * 2;
        s.host_kv_bytes = ((2 * q_bytes + 2 * s_bytes + 4095) / 4096) * 4096;
        void* hm = mmap(nullptr, s.host_kv_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (hm != MAP_FAILED && cudaHostRegister(hm, s.host_kv_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess) {
            s.host_kv_arena = (uint8_t*) hm;
            void* dp = nullptr;
            CHECK(cudaHostGetDevicePointer(&dp, hm, 0));
            uint8_t* dbase = (uint8_t*) dp;
            s.hk_q = (signed char*) dbase;
            s.hv_q = (signed char*) (dbase + q_bytes);
            s.hk_s = (unsigned short*) (dbase + 2 * q_bytes);
            s.hv_s = (unsigned short*) (dbase + 2 * q_bytes + s_bytes);
        } else {
            cudaGetLastError();
            if (hm != MAP_FAILED) munmap(hm, s.host_kv_bytes);
            std::fprintf(stderr, "[wzy-pp2] warning: host 256K KV pin declined on GPU%d, using VRAM ring\n", dev);
        }
    }

    // Multi-row [MAX_BATCH, ...] scratch + Unlocked Tensor-Core FP16 workspace
    s.R = alloc0((size_t) MAX_BATCH * HCD);
    s.R_pre_head = alloc0((size_t) MAX_BATCH * HCD);
    s.xn = alloc0((size_t) MAX_BATCH * HCD);
    s.lo = alloc0((size_t) MAX_BATCH * HCLR);
    s.mixed = alloc0((size_t) MAX_BATCH * NE);
    s.mixed2 = alloc0((size_t) MAX_BATCH * NE);
    s.inject = alloc0((size_t) MAX_BATCH * HC);
    s.block = alloc0((size_t) MAX_BATCH * NE);

    s.qkv = alloc0((size_t) MAX_BATCH * QKV);
    s.convout = alloc0((size_t) MAX_BATCH * QKV);
    s.z = alloc0((size_t) MAX_BATCH * GV);
    s.decay = alloc0((size_t) MAX_BATCH * VH);
    s.bt = alloc0((size_t) MAX_BATCH * VH);
    s.on = alloc0((size_t) MAX_BATCH * GV);

    s.qg = alloc0((size_t) MAX_BATCH * 2 * QD);
    s.kk = alloc0((size_t) MAX_BATCH * KVD);
    s.vv = alloc0((size_t) MAX_BATCH * KVD);
    s.attn = alloc0((size_t) MAX_BATCH * QD);
    s.idx_raw = alloc0((size_t) MAX_BATCH * IDX_DIM);
    s.q_idx = alloc0((size_t) MAX_BATCH * IDX_HEADS * IDX_DIM);

    s.slog = alloc0((size_t) MAX_BATCH * NEXP);
    s.sids = alloc0((size_t) MAX_BATCH * TOPK);
    s.swts = alloc0((size_t) MAX_BATCH * TOPK);
    s.res_swts = alloc0((size_t) MAX_BATCH * TOPK);
    s.sff = alloc0((size_t) MAX_BATCH * NFF);
    s.sup = alloc0((size_t) MAX_BATCH * NFF);
    s.tmp_ff = alloc0((size_t) MAX_BATCH * NFF);
    s.sshared = alloc0((size_t) MAX_BATCH * NE);
    s.sgate = alloc0((size_t) MAX_BATCH);

    s.parts = alloc0((size_t) MAX_BATCH * TOPK * NE);
    s.comb = alloc0((size_t) MAX_BATCH * NE);
    s.ffact = alloc0((size_t) MAX_BATCH * TOPK * NFF);
    s.exsc = alloc0((size_t) MAX_BATCH * TOPK * 3);

    CHECK(cudaMalloc(&s.w_fp16, (size_t) 12288 * NE * sizeof(__half))); // 60 MiB (headq stays on gemm_q8p since head only runs on T <= MAX_W!)
    CHECK(cudaMalloc(&s.x_fp16, (size_t) MAX_BATCH * HCD * sizeof(__half)));

    const size_t AMAX = (size_t) MAX_BATCH * HCD;
    CHECK(cudaMalloc(&s.xq8, AMAX));
    CHECK(cudaMalloc(&s.xs8, (AMAX / 32 + 16) * 4));
    CHECK(cudaMalloc(&s.eq32, (size_t) MAX_BATCH * NE));
    CHECK(cudaMalloc(&s.es32, (size_t) (MAX_BATCH * NE / 32 + 16) * 4));
    CHECK(cudaMalloc(&s.eq16, (size_t) MAX_BATCH * NE));
    CHECK(cudaMalloc(&s.es16, (size_t) (MAX_BATCH * NE / 16 + 16) * 4));
    CHECK(cudaMalloc(&s.dq16, (size_t) MAX_BATCH * TOPK * NFF));
    CHECK(cudaMalloc(&s.ds16, (size_t) (MAX_BATCH * TOPK * NFF / 16 + 16) * 4));

    const int MAX_K = MAX_BATCH * TOPK;
    CHECK(cudaMalloc(&s.grp_slot, MAX_K * 4));
    CHECK(cudaMalloc(&s.grp_cnt, MAX_K * 4));
    CHECK(cudaMalloc(&s.grp_e, MAX_K * GRP_MAX * 4));
    CHECK(cudaMalloc(&s.d_ngroups, 4));

    if (s.has_ple) {
        s.ple_wk = up_bf(g, "blk.1.ple_key.weight");
        s.ple_wv = up_bf(g, "blk.1.ple_value.weight");
        s.ple_nk = up_f32(g, "blk.1.ple_norm_key.weight");
        s.ple_nq = up_f32(g, "blk.1.ple_norm_query.weight");
        s.ple_nc = up_f32(g, "blk.1.ple_norm_conv.weight");
        s.ple_cv = up_f32(g, "blk.1.ple_conv1d.weight");
        s.ple_hist = alloc0((size_t) HCD * PLE_HIST);
        s.ple_snaps = alloc0((size_t) MAX_W * HCD * PLE_HIST);
        s.d_ple_emb = alloc0((size_t) MAX_BATCH * NE);
        s.ple_q = alloc0((size_t) MAX_BATCH * HCD);
        s.ple_k = alloc0((size_t) MAX_BATCH * HCD);
        s.ple_v = alloc0((size_t) MAX_BATCH * NE);
        s.ple_gat = alloc0((size_t) MAX_BATCH * HCD);
        s.ple_n = alloc0((size_t) MAX_BATCH * HCD);
    }

    if (s.has_head) {
        s.headnorm = up_f32(g, "output_hc_norm.weight");
        s.head_down = up_bf_as_q8(g, "output_hc_down.weight");
        s.head_up = up_bf_as_q8(g, "output_hc_up.weight");
        s.headq = up_q8(g, "output.weight");
        s.hlogits = alloc0((size_t) MAX_W * NVOCAB);
        CHECK(cudaHostAlloc(&s.h_greedy_ids, MAX_W * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        CHECK(cudaHostGetDevicePointer(&s.d_greedy_ids, s.h_greedy_ids, 0));
        CHECK(cudaHostAlloc(&s.h_top16_val, MAX_W * 16 * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
        CHECK(cudaHostGetDevicePointer(&s.d_top16_val, s.h_top16_val, 0));
        CHECK(cudaHostAlloc(&s.h_top16_id, MAX_W * 16 * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        CHECK(cudaHostGetDevicePointer(&s.d_top16_id, s.h_top16_id, 0));
    }

    CHECK(cudaMalloc(&s.dpos, 4));
    CHECK(cudaMalloc(&s.dmisses, 4));
    CHECK(cudaMalloc(&s.d_hits, 4));
    CHECK(cudaMemset(s.d_hits, 0, 4));
    CHECK(cudaMalloc(&s.dslots, MAX_K * 4));
    CHECK(cudaMalloc(&s.mkey, 2048 * 4));
    s.h_mkey.resize(2048);
    CHECK(cudaMalloc(&s.d_lut, 48 * NEXP * 4));
    CHECK(cudaHostAlloc(&s.h_lut_pinned, 48 * NEXP * sizeof(int), cudaHostAllocPortable));
    s.h_lut.assign(48 * NEXP, -1);
    for (int i = 0; i < 48 * NEXP; ++i) s.h_lut_pinned[i] = -1;
    s.cold_freq.assign(48 * NEXP, 0);
    s.pending_promo.reserve(32);
    CHECK(cudaHostAlloc(&s.h_adapt_ref, 8192, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(s.h_adapt_ref, 0, 8192);
    CHECK(cudaHostGetDevicePointer(&s.d_adapt_ref, s.h_adapt_ref, 0));
    CHECK(cudaHostAlloc(&s.h_route_freq, 48 * NEXP * sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(s.h_route_freq, 0, 48 * NEXP * sizeof(uint32_t));
    CHECK(cudaHostGetDevicePointer(&s.d_route_freq, s.h_route_freq, 0));
    CHECK(cudaHostAlloc((void**) &s.h_skip_flag, 64, cudaHostAllocMapped | cudaHostAllocPortable));
    *s.h_skip_flag = 0;
    CHECK(cudaHostGetDevicePointer(&s.d_skip_flag, (void*) s.h_skip_flag, 0));
    CHECK(cudaHostAlloc(&s.h_cold_req, 48 * sizeof(ColdLayerReq), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset((void*) s.h_cold_req, 0, 48 * sizeof(ColdLayerReq));
    CHECK(cudaHostGetDevicePointer(&s.d_cold_req, (void*) s.h_cold_req, 0));
    for (int b = s.l_begin; b < s.l_end; ++b) {
        CHECK(cudaHostAlloc(&s.h_cold_out[b], (size_t) MAX_W * NE * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(s.h_cold_out[b], 0, (size_t) MAX_W * NE * sizeof(float));
        CHECK(cudaHostGetDevicePointer(&s.d_cold_out[b], s.h_cold_out[b], 0));
    }
}

static void alloc_stage_expert_vram(StageGpu& s) {
    CHECK(cudaSetDevice(s.dev));
    const size_t reserve_mib = std::getenv("WZY_RESERVE_MIB") ? std::atol(std::getenv("WZY_RESERVE_MIB")) : 150;
    // Batch engine: hold VRAM back for N resident session slots (KV ring + GDN state + indexer pool).
    // Experts must not swallow it, otherwise concurrency needs cache eviction to make room.
    const size_t leave_mib = std::getenv("WZY_LEAVE_MIB") ? std::atol(std::getenv("WZY_LEAVE_MIB")) : 0;
    size_t free_b = 0, total_b = 0;
    CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t res_b = (reserve_mib + leave_mib) * 1024ull * 1024ull;
    const size_t budget = free_b > res_b + 64 * PER ? (free_b - res_b) : (free_b / 2);
    const int max_stage_experts = (s.l_end - s.l_begin) * NEXP;
    // WZY_EXPERT_CAP pins the slot count. Only the A/B parity gate needs it: two engines with
    // different free VRAM would otherwise pick different hot/cold splits, and cold experts are
    // computed on the CPU (different rounding) so their token streams would legitimately differ.
    const int cap_env = std::getenv("WZY_EXPERT_CAP") ? std::atoi(std::getenv("WZY_EXPERT_CAP")) : 0;
    s.cap = cap_env > 0 ? std::min(cap_env, max_stage_experts)
                        : std::min((int) (budget / PER), max_stage_experts);
    // The static tier is frozen from --expert-profile; only the adaptive tier can follow the live workload.
    // A profile that mismatches the session therefore pins VRAM to useless experts (measured: 80% vs 94% hit on
    // Chinese compliance text). Keep a smaller frozen core and let most of VRAM adapt (WZY_CORE_KEEP / WZY_ADAPT_SLOTS).
    const int core_keep = std::getenv("WZY_CORE_KEEP") ? std::max(1024, std::atoi(std::getenv("WZY_CORE_KEEP"))) : 3840;
    const int adapt_want = std::getenv("WZY_ADAPT_SLOTS") ? std::max(0, std::atoi(std::getenv("WZY_ADAPT_SLOTS"))) : 2048;
    s.adapt_cap = std::min(std::max(0, s.cap - core_keep), std::min(6144, adapt_want));
    s.static_cap = s.cap - s.adapt_cap;
    s.ring_owner.assign(std::max(1, s.adapt_cap), -1);
    CHECK(cudaMalloc(&s.xblobs, (size_t) s.cap * PER));
    CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::fprintf(stderr, "[wzy-v2] GPU%d layers [%d..%d): %d VRAM expert slots (%d static + %d adaptive, %.2f GiB), %.0f MiB free\n",
                 s.dev, s.l_begin, s.l_end, s.cap, s.static_cap, s.adapt_cap,
                 (double) s.cap * PER / 1073741824.0, free_b / 1048576.0);
}

static void parse_json_ints(const std::string& line, const char* key, std::vector<int>& out) {
    out.clear();
    const size_t p = line.find(key);
    if (p == std::string::npos) return;
    const size_t b = line.find('[', p);
    const size_t e = line.find(']', b);
    if (b == std::string::npos || e == std::string::npos) return;
    int cur = 0; bool neg = false, any = false;
    for (size_t i = b + 1; i < e; ++i) {
        const char ch = line[i];
        if (ch == '-') neg = true;
        else if (ch >= '0' && ch <= '9') { cur = cur * 10 + (ch - '0'); any = true; }
        else {
            if (any) out.push_back(neg ? -cur : cur);
            cur = 0; neg = false; any = false;
        }
    }
    if (any) out.push_back(neg ? -cur : cur);
}

static double parse_json_num(const std::string& line, const char* key, double def) {
    const size_t p = line.find(key);
    if (p == std::string::npos) return def;
    const size_t c = line.find(':', p);
    if (c == std::string::npos) return def;
    size_t i = c + 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    char* endp = nullptr;
    const double v = std::strtod(line.c_str() + i, &endp);
    return (endp == line.c_str() + i) ? def : v;
}

// Minimal JSON string reader (`"key":"value"`), escape-aware. Used for the optional vision fields.
static bool parse_json_str(const std::string& line, const char* key, std::string& out) {
    const size_t p = line.find(key);
    if (p == std::string::npos) return false;
    size_t i = line.find('"', p + std::strlen(key));
    if (i == std::string::npos) return false;
    std::string s;
    for (++i; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\\' && i + 1 < line.size()) { s.push_back(line[++i]); continue; }
        if (c == '"') { out = s; return true; }
        s.push_back(c);
    }
    return false;
}

