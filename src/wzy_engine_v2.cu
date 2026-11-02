// wzy_engine_v2.cu -- Unified PP2 Engine:
//   1. Fused & Vectorized Multi-Row (W=1..8) Compute Backbone (Zero FP64)
//   2. 262,144 (256K) Context QSA 4-Head Sparse Indexer (top_k=2048) + INT8 Tiered KV Cache
//   3. Unlocked SM75 Tensor-Core (183 TFLOPS cublasGemmEx + Grouped Experts) Batched Chunked Prefill
//   4. Native MTP Speculative Decoding (<MODEL>/mtp Q2_0 Resident Experts + 114,452 CJK Draft Head)
#include "wzy_mtp.cuh"
#include "wzy_prefill_tc.cuh"
#include "wzy_suffix.cuh"
#include <csignal>
#include <cstring>
#include <fstream>
#include <algorithm>

static volatile sig_atomic_t g_cancel_req = 0;
static void on_sigusr1_cancel(int) { g_cancel_req = 1; }

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
                                     int draft_tok, int& out_tok, std::mt19937& rng) {
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
    size_t free_b = 0, total_b = 0;
    CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t res_b = reserve_mib * 1024ull * 1024ull;
    const size_t budget = free_b > res_b + 64 * PER ? (free_b - res_b) : (free_b / 2);
    const int max_stage_experts = (s.l_end - s.l_begin) * NEXP;
    s.cap = std::min((int) (budget / PER), max_stage_experts);
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

int main() {
    const auto t_boot0 = std::chrono::steady_clock::now();
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    const bool pp2 = (ndev >= 2) && (std::getenv("WZY_SINGLE") == nullptr);
    const int split_layer = pp2 ? (std::getenv("WZY_SPLIT") ? std::atoi(std::getenv("WZY_SPLIT")) : 25) : 48;
    const int ctx_max = std::getenv("WZY_CTX") ? std::atoi(std::getenv("WZY_CTX")) : MAX_CTX_256K;
    const bool explicit_pf_chunk = (std::getenv("WZY_PREFILL_CHUNK") != nullptr);
    const int prefill_chunk = std::min(PF_MAX, explicit_pf_chunk ? std::atoi(std::getenv("WZY_PREFILL_CHUNK")) : 25600);
    const int split_pf_chunks = std::getenv("WZY_SPLIT_PF_CHUNKS") ? std::max(1, std::atoi(std::getenv("WZY_SPLIT_PF_CHUNKS"))) : 2;

    if (pp2) {
        CHECK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1, 0); cudaGetLastError();
        CHECK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0, 0); cudaGetLastError();
        CHECK(cudaSetDevice(0));
    }

    Gguf g;
    std::string err;
    if (!g.open((std::string(KD) + "/qwen-nvfp4-gptq-dense.gguf").c_str(), err)) {
        std::fprintf(stderr, "gguf error: %s\n", err.c_str());
        return 2;
    }
    std::vector<float> h_cos((size_t) ctx_max * 32), h_sin((size_t) ctx_max * 32);
    build_rope_cos_sin_host(ctx_max, ROPE_THETA, h_cos.data(), h_sin.data());

    const int nstages = pp2 ? 2 : 1;
    std::vector<StageGpu> stages(nstages);
    if (pp2) {
        init_stage_weights_and_state(stages[0], 0, 0, split_layer, ctx_max, g, h_cos, h_sin);
        init_stage_weights_and_state(stages[1], 1, split_layer, 48, ctx_max, g, h_cos, h_sin);
    } else {
        init_stage_weights_and_state(stages[0], 0, 0, 48, ctx_max, g, h_cos, h_sin);
    }

    // Load MTP Speculative Drafter on the last GPU stage BEFORE allocating expert caches so remaining VRAM goes cleanly to experts!
    MtpEngine mtp;
    if (std::getenv("WZY_NOMTP") == nullptr) {
        std::string mtp_dir = std::string(KD) + "/mtp";
        std::string dv_path = std::getenv("WZY_DRAFT_VOCAB")
            ? std::getenv("WZY_DRAFT_VOCAB")
            : "data/draft_vocab_han.bin";
        if (!std::ifstream(dv_path)) dv_path = "/home/liaoyang/wzy-flash-serve/data/draft_vocab_han.bin";
        if (!std::ifstream(dv_path)) dv_path = "/home/liaoyang/layftllm/strata-fork/data/draft_vocab_han.bin";
        if (!std::ifstream(dv_path)) dv_path = mtp_dir + "/draft_vocab.bin";
        mtp.init(stages.back().dev, g, mtp_dir, dv_path, stages.back().stream);
    }

    for (int si = 0; si < nstages; ++si) alloc_stage_expert_vram(stages[si]);

    // Map & UVA-register experts.bin (exact 256*PER = 675 MiB page-and-blob aligned chunks)
    const std::string exp_path = std::string(KD) + "/pack/experts.bin";
    const int efd = ::open(exp_path.c_str(), O_RDONLY);
    if (efd < 0) { std::fprintf(stderr, "cannot open %s\n", exp_path.c_str()); return 2; }
    struct stat est{};
    fstat(efd, &est);
    const size_t exp_sz = (size_t) est.st_size;
    uint8_t* host_pack = (uint8_t*) mmap(nullptr, exp_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE, efd, 0);
    ::close(efd);
    if (host_pack == MAP_FAILED) { std::fprintf(stderr, "mmap experts.bin failed\n"); return 2; }

    if (std::getenv("WZY_NOPIN") == nullptr) {
        std::fprintf(stderr, "[wzy-v2] Pre-faulting & pinning %.2f GiB expert pack across 16 threads...\n", exp_sz / 1073741824.0);
        const int nth = 16;
        std::vector<std::thread> workers;
        for (int tid = 0; tid < nth; ++tid) {
            workers.emplace_back([&, tid]() {
                const size_t pages = exp_sz / 4096;
                volatile uint8_t sink = 0;
                for (size_t pg = tid; pg < pages; pg += nth) sink ^= host_pack[pg * 4096];
                (void) sink;
            });
        }
        for (auto& th : workers) th.join();
    }
    {
        const size_t CHUNK = (size_t) 256 * PER;
        if (cudaHostRegister(host_pack, exp_sz, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            cudaGetLastError();
            for (size_t off = 0; off < exp_sz; off += CHUNK) {
                const size_t len = std::min(CHUNK, exp_sz - off);
                cudaHostRegister(host_pack + off, len, cudaHostRegisterPortable | cudaHostRegisterMapped);
                cudaGetLastError();
            }
        }
    }
    for (int si = 0; si < nstages; ++si) {
        CHECK(cudaSetDevice(stages[si].dev));
        void* dp = nullptr;
        CHECK(cudaHostGetDevicePointer(&dp, host_pack, 0));
        stages[si].d_pack = (uint8_t*) dp;
    }

    // Prefill all VRAM expert slots from expert-profile-cjk.bin (or fallback to expert-profile.bin)
    std::string prof_path = std::getenv("WZY_PROFILE")
        ? std::getenv("WZY_PROFILE")
        : "expert-profile-cjk.bin";
    if (!std::ifstream(prof_path)) prof_path = "data/expert-profile-cjk.bin";
    if (!std::ifstream(prof_path)) prof_path = "/home/liaoyang/wzy-flash-serve/expert-profile-cjk.bin";
    const bool using_cjk_prof = (bool) std::ifstream(prof_path);
    if (!using_cjk_prof) {
        prof_path = "data/expert-profile.bin";
        if (!std::ifstream(prof_path)) prof_path = "/home/liaoyang/layftllm/strata-fork/data/expert-profile.bin";
    }
    std::vector<std::pair<int, int>> prior_ranked;
    {
        std::ifstream pf(prof_path, std::ios::binary);
        std::vector<std::pair<int, int>> ranked;
        if (pf) {
            pf.seekg(0, std::ios::end);
            const size_t psz = (size_t) pf.tellg();
            pf.seekg(0);
            std::vector<uint8_t> raw(psz);
            pf.read((char*) raw.data(), psz);
            if (psz >= 24 && std::memcmp(raw.data(), "STRP", 4) == 0) {
                uint32_t hdr[5];
                std::memcpy(hdr, raw.data() + 4, 20);
                const uint32_t nl = hdr[1], ne = hdr[2], nrank = hdr[4];
                for (uint32_t i = 0; i < nrank && 24 + (size_t) (i + 1) * 4 <= psz; ++i) {
                    uint16_t pair[2];
                    std::memcpy(pair, raw.data() + 24 + (size_t) i * 4, 4);
                    if (pair[0] < nl && pair[1] < ne) ranked.emplace_back((int) pair[0], (int) pair[1]);
                }
            }
        }
        for (int l = 0; l < 48; ++l)
            for (int e = 0; e < NEXP; ++e)
                ranked.emplace_back(l, e);
        prior_ranked = ranked;
        for (int si = 0; si < nstages; ++si) {
            StageGpu& stg = stages[si];
            CHECK(cudaSetDevice(stg.dev));
            int filled = 0;
            for (const auto& pr : ranked) {
                if (filled >= stg.cap) break;
                const int l = pr.first, e = pr.second;
                if (l < stg.l_begin || l >= stg.l_end) continue;
                const int gkey = l * NEXP + e;
                if (stg.h_lut[gkey] >= 0) continue;
                const int vslot = filled++;
                stg.h_lut[gkey] = vslot;
                stg.h_lut_pinned[gkey] = vslot;
                if (vslot >= stg.static_cap) stg.ring_owner[vslot - stg.static_cap] = gkey;
                CHECK(cudaMemcpyAsync(stg.xblobs + (size_t) vslot * PER,
                                      host_pack + (size_t) gkey * PER,
                                      PER, cudaMemcpyHostToDevice, stg.stream));
            }
            if (using_cjk_prof && stg.h_adapt_ref != nullptr && stg.adapt_cap > 0) {
                std::memset(stg.h_adapt_ref, 1, (size_t) stg.adapt_cap);
            }
            CHECK(cudaMemcpyAsync(stg.d_lut, stg.h_lut.data(), 48 * NEXP * 4, cudaMemcpyHostToDevice, stg.stream));
            CHECK(cudaStreamSynchronize(stg.stream));
            std::fprintf(stderr, "[wzy-v2] GPU%d prefilled %d VRAM expert slots (profile=%s)\n", stg.dev, filled, prof_path.c_str());
        }
    }
    std::vector<PrefillStageCtx> pf_ctx(nstages);
    for (int si = 0; si < nstages; ++si) pf_ctx[si].init(stages[si]);

    // BF16 Token Embedding
    const TDesc* te = g.find("token_embd.weight");
    const uint8_t* embbase = g.at(te);
    const int embnb = NE / 32;
    const uint16_t* bf_emb_base = nullptr;
    {
        const int bfd = ::open((std::string(KD) + "/token-embd-bf16.gguf").c_str(), O_RDONLY);
        if (bfd >= 0) {
            struct stat bst{};
            if (fstat(bfd, &bst) == 0 && (size_t) bst.st_size >= 256 + (size_t) NVOCAB * NE * 2) {
                uint8_t* em = (uint8_t*) mmap(nullptr, (size_t) bst.st_size, PROT_READ, MAP_PRIVATE, bfd, 0);
                if (em != MAP_FAILED && std::memcmp(em, "GGUF", 4) == 0) {
                    madvise(em, (size_t) bst.st_size, MADV_WILLNEED);
                    volatile uint8_t sink_em = 0;
                    for (size_t off = 0; off < (size_t) bst.st_size; off += 4096) sink_em ^= em[off];
                    (void) sink_em;
                    bf_emb_base = (const uint16_t*) (em + 256);
                }
            }
            ::close(bfd);
        }
    }

    // Register the BF16 embedding mmap as mapped host (UVA) memory so the MTP draft chain can gather
    // embeddings on device. This costs no VRAM (a device-side table would need 1.27 GiB, only ~0.55 free).
    const uint16_t* d_emb_uva = nullptr;
    const bool emb_uva_on = (std::getenv("WZY_EMB_UVA") != nullptr);
    if (bf_emb_base != nullptr && emb_uva_on) {
        void* reg_base = (void*) ((uint8_t*) bf_emb_base - 256);
        const size_t reg_len = 256 + (size_t) NVOCAB * NE * 2;
        const cudaError_t reg_err = cudaHostRegister(reg_base, reg_len,
                                                     cudaHostRegisterReadOnly | cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (reg_err != cudaSuccess) {
            std::fprintf(stderr, "[wzy-v2] WARN: token-embd-bf16 UVA register failed: %s (len=%zu MiB)%c",
                         cudaGetErrorString(reg_err), reg_len >> 20, 10);
        } else {
            void* dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, (void*) bf_emb_base, 0) == cudaSuccess) {
                d_emb_uva = (const uint16_t*) dp;
                std::fprintf(stderr, "[wzy-v2] token-embd-bf16 registered UVA (%zu MiB, no VRAM); device emb gather ON\n", reg_len >> 20);
            } else {
                cudaGetLastError();
            }
        }
    }
    if (d_emb_uva != nullptr) {
        if (mtp.enabled) mtp.set_emb_table(d_emb_uva);
        CHECK(cudaSetDevice(stages[0].dev));
        stages[0].set_emb_dev(d_emb_uva);
        CHECK(cudaMalloc(&stages[0].d_toks, (size_t) MAX_BATCH * 4));
        std::fprintf(stderr, "[wzy-v2] stage0 first-layer embedding via device gather ON%c", 10);
    }

    CHECK(cudaSetDevice(0));
    float *h_emb = nullptr, *h_ple = nullptr, *h_logits = nullptr;
    int* h_pos = nullptr;
    CHECK(cudaHostAlloc(&h_emb, (size_t) MAX_BATCH * NE * 4, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_ple, (size_t) MAX_BATCH * NE * 4, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_pos, 8, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_logits, (size_t) MAX_W * NVOCAB * 4, cudaHostAllocPortable));

    auto stage_one_emb = [&](float* dst, int id) {
        if (id < 0 || id >= NVOCAB) id = 0;
        if (bf_emb_base != nullptr) {
            const uint16_t* rbf = bf_emb_base + (size_t) id * NE;
            for (int i = 0; i < NE; ++i) dst[i] = bf2f(rbf[i]);
        } else {
            const uint8_t* row = embbase + (size_t) id * embnb * 34;
            for (int b = 0; b < embnb; ++b) {
                const float d = h2f(*(const uint16_t*) (row + b * 34));
                const int8_t* qp = (const int8_t*) (row + b * 34 + 2);
                for (int i = 0; i < 32; ++i) dst[b * 32 + i] = d * qp[i];
            }
        }
    };

    // Boot self-test (WZY_EMB_TEST=1): device UVA gather must equal the host path bit for bit.
    if (std::getenv("WZY_EMB_TEST") && mtp.enabled && d_emb_uva != nullptr) {
        int* d_tok = nullptr;
        float* d_out = nullptr;
        CHECK(cudaMalloc(&d_tok, 4));
        CHECK(cudaMalloc(&d_out, (size_t) NE * 4));
        std::vector<float> got(NE), want(NE);
        const int probes[] = {0, 760, 1000, 15321, 50000, 100000, 200000, 248319};
        float worst = 0.f;
        int worst_tok = -1;
        for (int id : probes) {
            CHECK(cudaMemcpy(d_tok, &id, 4, cudaMemcpyHostToDevice));
            dim3 g((NE + 255) / 256, 1);
            mtp_gather_bf16_emb<<<g, 256>>>(d_emb_uva, d_tok, d_out, 1);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(got.data(), d_out, (size_t) NE * 4, cudaMemcpyDeviceToHost));
            stage_one_emb(want.data(), id);
            float m = 0.f;
            for (int i = 0; i < NE; ++i) m = std::max(m, std::fabs(got[i] - want[i]));
            if (m > worst) { worst = m; worst_tok = id; }
        }
        CHECK(cudaFree(d_tok));
        CHECK(cudaFree(d_out));
        std::fprintf(stderr, "[wzy-emb-test] device UVA gather vs host path: max|diff| = %.9g (token %d), probes=%zu -> %s\n",
                     worst, worst_tok, sizeof(probes) / sizeof(probes[0]), worst == 0.f ? "EXACT" : "MISMATCH");
    }

    PleSrc plesrc;
    if (!plesrc.open((std::string(KD) + "/ple-fp8.gguf").c_str())) {
        std::fprintf(stderr, "cannot map ple-fp8.gguf\n");
        return 2;
    }
    static const uint64_t ple_mult[3] = {23703573157769ull, 20109073645365ull, 8052911324071ull};
    static const uint64_t ple_vocab[PLE_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    static const uint64_t ple_off[PLE_HEADS] = {
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    int ple_prev[2] = {-1, -1};

    auto warm_ple_token = [&](int cur, int& p0, int& p1) {
        const int32_t t3[3] = {p0, p1, cur};
        int64_t ctx[PLE_NG];
        ctx[0] = cur;
        bool cut = false;
        for (int s = 1; s < PLE_NG; ++s) {
            const int32_t v = cut ? -1 : t3[2 - s];
            cut = cut || v < 0 || v == PLE_EOS;
            ctx[s] = cut ? PLE_EOS : v;
        }
        uint32_t r_all[PLE_HEADS];
        for (int n = 2; n <= PLE_NG; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * ple_mult[j];
            for (int gi = 0; gi < PLE_HPN; ++gi) {
                const int h = (n - 2) * PLE_HPN + gi;
                const uint32_t r = (uint32_t) (mixed % ple_vocab[h] + ple_off[h]);
                r_all[h] = r;
                plesrc.prefetch(r);
            }
        }
        for (int h = 0; h < PLE_HEADS; ++h) plesrc.warm_row(r_all[h]);
        p0 = p1; p1 = cur;
    };

    auto stage_ple_window = [&](const int* toks, int T, bool advance_state, float* ple_buf = nullptr, const int* prev_ov = nullptr) {
        int p0 = prev_ov ? prev_ov[0] : ple_prev[0];
        int p1 = prev_ov ? prev_ov[1] : ple_prev[1];
        float* base_dst = ple_buf ? ple_buf : h_ple;
        uint32_t r_win[MAX_BATCH][PLE_HEADS];
        for (int t = 0; t < T; ++t) {
            const int cur = toks[t];
            const int32_t t3[3] = {p0, p1, cur};
            int64_t ctx[PLE_NG];
            ctx[0] = cur;
            bool cut = false;
            for (int s = 1; s < PLE_NG; ++s) {
                const int32_t v = cut ? -1 : t3[2 - s];
                cut = cut || v < 0 || v == PLE_EOS;
                ctx[s] = cut ? PLE_EOS : v;
            }
            for (int n = 2; n <= PLE_NG; ++n) {
                uint64_t mixed = (uint64_t) ctx[0] * ple_mult[0];
                for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * ple_mult[j];
                for (int gi = 0; gi < PLE_HPN; ++gi) {
                    const int h = (n - 2) * PLE_HPN + gi;
                    const uint32_t r = (uint32_t) (mixed % ple_vocab[h] + ple_off[h]);
                    r_win[t][h] = r;
                    plesrc.prefetch(r);
                }
            }
            p0 = p1; p1 = cur;
        }
        for (int t = 0; t < T; ++t) {
            float* dst = base_dst + (size_t) t * NE;
            for (int h = 0; h < PLE_HEADS; ++h) {
                plesrc.row(r_win[t][h], dst + (size_t) h * PLE_HEAD);
            }
        }
        if (advance_state) { ple_prev[0] = p0; ple_prev[1] = p1; }
    };
    auto advance_ple_prev = [&](const int* toks, int T) {
        if (T >= 2) { ple_prev[0] = toks[T - 2]; ple_prev[1] = toks[T - 1]; }
        else if (T == 1) { ple_prev[0] = ple_prev[1]; ple_prev[1] = toks[0]; }
    };

    auto reset_all = [&]() {
        for (auto& stg : stages) stg.reset_state();
        if (mtp.enabled) mtp.reset_kv();
        ple_prev[0] = -1;
        ple_prev[1] = -1;
    };

    const int cpu_nth = std::getenv("WZY_CPU_THREADS") ? std::max(1, std::atoi(std::getenv("WZY_CPU_THREADS"))) : 12;
    CpuMoePool cpu_pool;
    cpu_pool.init(cpu_nth);

    *h_pos = 0;
    for (int t = 0; t < MAX_W; ++t) stage_one_emb(h_emb + (size_t) t * NE, 760);
    int dummy_toks[MAX_W];
    for (int t = 0; t < MAX_W; ++t) dummy_toks[t] = 760;
    stage_ple_window(dummy_toks, MAX_W, false);

    for (int si = 0; si < nstages; ++si) {
        StageGpu& stg = stages[si];
        CHECK(cudaSetDevice(stg.dev));
        const uint32_t bseq = stg.cold_seq;
        stg.record_forward(h_pos, h_emb, h_ple, 1, stg.has_head, false, host_pack, &cpu_pool);
        stg.service_cold_layers(host_pack, &cpu_pool, bseq, 1);
        CHECK(cudaStreamSynchronize(stg.stream));
        stg.reset_state();
    }
    if (mtp.enabled) {
        stage_one_emb(mtp.h_emb_stage, 760);
        mtp.capture_step1(stages.back().cos_tab, stages.back().sin_tab);
    }
    reset_all();
    for (auto& stg : stages) { stg.total_lookups = 0; stg.total_vram_hits = 0; stg.total_promotions = 0; }

    cudaEvent_t ev_s0_done = nullptr;
    if (pp2) {
        CHECK(cudaSetDevice(stages[0].dev));
        CHECK(cudaEventCreateWithFlags(&ev_s0_done, cudaEventDisableTiming));
    }

    struct S1AsyncJob {
        int W = 1;
        bool need_logits = false;
        bool save_snaps = false;
        bool fast_greedy = false;
        bool want_top16 = false;
        uint32_t s1_base = 0;
    } s1_job;
    alignas(64) std::atomic<uint32_t> s1_req_seq{0};
    alignas(64) std::atomic<uint32_t> s1_done_seq{0};
    std::atomic<bool> s1_stop{false};
    std::thread s1_enq_worker;
    if (pp2) {
        s1_enq_worker = std::thread([&]() {
            cpu_pool.s1_launcher_tid.store((int) syscall(SYS_gettid), std::memory_order_release);
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(14, &cs);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cs);
            CHECK(cudaSetDevice(stages[1].dev));
            uint32_t last_seq = 0;
            while (!s1_stop.load(std::memory_order_acquire)) {
                uint32_t seq = s1_req_seq.load(std::memory_order_acquire);
                if (seq == last_seq) {
                    if (cpu_pool.sleep_mode.load(std::memory_order_relaxed)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(250));
                    } else {
                        _mm_pause();
                    }
                    continue;
                }
                if (s1_stop.load(std::memory_order_acquire)) break;
                last_seq = seq;
                const int W = s1_job.W;
                CHECK(cudaStreamWaitEvent(stages[1].stream, ev_s0_done, 0));
                CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev,
                                          stages[0].R.p, stages[0].dev,
                                          (size_t) W * HCD * 4, stages[1].stream));
                s1_job.s1_base = stages[1].cold_seq;
                stages[1].record_forward(h_pos, h_emb, h_ple, W, s1_job.need_logits, s1_job.save_snaps,
                                         host_pack, &cpu_pool, s1_job.want_top16, 0, false);
                if (s1_job.need_logits && !s1_job.fast_greedy && !s1_job.want_top16) {
                    CHECK(cudaMemcpyAsync(h_logits, stages[1].hlogits.p, (size_t) W * NVOCAB * 4,
                                          cudaMemcpyDeviceToHost, stages[1].stream));
                }
                s1_done_seq.store(seq, std::memory_order_release);
            }
        });
        while (cpu_pool.s1_launcher_tid.load(std::memory_order_acquire) < 0) _mm_pause();
    }
    cpu_pool.sweep_aux_threads();

    const double boot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_boot0).count();
    int total_vram_slots = 0;
    for (const auto& stg : stages) total_vram_slots += stg.cap;
    double prof_enq_ms = 0.0, prof_s0_ms = 0.0, prof_s1_ms = 0.0, prof_mtp_ms = 0.0;
    int prof_wins = 0;
    // ---- Vision (optional): rows produced by `strata-vision` (file: int32 {SVE1,n_tokens,nx,ny,n_embd}
    // then f32 [n_tokens][n_embd], one block per image in prompt order).  The caller expands each image's single
    // `<|image_pad|>` into `count[k]` pads and reports where that run starts; the rows replace the text
    // embeddings of those tokens. ----
    std::vector<float> img_emb;
    std::vector<int> img_run_start, img_run_count;
    auto load_img_emb = [&](const std::string& path, const std::vector<int>& starts, const std::vector<int>& counts) {
        img_emb.clear();
        img_run_start.clear();
        img_run_count.clear();
        if (path.empty() || starts.empty() || starts.size() != counts.size()) return;
        long total = 0;
        for (int c : counts) {
            if (c <= 0) return;
            total += c;
        }
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "[wzy-vision] cannot open %s%c", path.c_str(), 10);
            return;
        }
        int32_t hdr[5] = {0, 0, 0, 0, 0};
        f.read((char*) hdr, sizeof hdr);
        if (hdr[0] != 0x31455653 || hdr[1] < total || hdr[4] != NE) {
            std::fprintf(stderr, "[wzy-vision] bad header magic=%08x rows=%d dim=%d (want rows>=%ld dim=%d)%c",
                         hdr[0], hdr[1], hdr[4], total, NE, 10);
            return;
        }
        img_emb.resize((size_t) total * NE);
        f.read((char*) img_emb.data(), (std::streamsize) ((size_t) total * NE * sizeof(float)));
        if (!f) {
            img_emb.clear();
            std::fprintf(stderr, "[wzy-vision] short read%c", 10);
            return;
        }
        img_run_start = starts;
        img_run_count = counts;
        std::fprintf(stderr, "[wzy-vision] %ld x %d rows over %zu run(s):", total, NE, starts.size());
        for (size_t k = 0; k < starts.size(); ++k) std::fprintf(stderr, " %d+%d", starts[k], counts[k]);
        std::fprintf(stderr, "%c", 10);
    };
    // Overwrite the rows of `dst_emb` (tokens p_start..p_start+B) that fall inside any image run.
    auto inject_img_emb = [&](float* dst_emb, int p_start, int B) {
        if (img_run_start.empty()) return;
        long off = 0;
        for (size_t k = 0; k < img_run_start.size(); ++k) {
            const int s = img_run_start[k], c = img_run_count[k];
            const int lo = std::max(p_start, s);
            const int hi = std::min(p_start + B, s + c);
            for (int gi = lo; gi < hi; ++gi) {
                std::memcpy(dst_emb + (size_t) (gi - p_start) * NE,
                            img_emb.data() + (size_t) (off + (gi - s)) * NE,
                            (size_t) NE * sizeof(float));
            }
            off += c;
        }
    };
    auto run_verify_window = [&](const int* toks, int W, int base_pos, bool need_logits, bool promote,
                                 bool fast_greedy = false, bool want_top16 = false, bool is_prefill_step = false) {
        const auto tw0 = std::chrono::steady_clock::now();
        *h_pos = base_pos;
        for (int t = 0; t < W; ++t) stage_one_emb(h_emb + (size_t) t * NE, toks[t]);
        inject_img_emb(h_emb, base_pos, W);
        stage_ple_window(toks, W, is_prefill_step || W == 1);
        const bool save_snaps = (!is_prefill_step && W > 1);

        CHECK(cudaSetDevice(stages[0].dev));
        const uint32_t s0_base = stages[0].cold_seq;
        int l0_served = stages[0].l_begin;
        stages[0].record_forward(h_pos, h_emb, h_ple, W, !pp2 && need_logits, save_snaps, host_pack, &cpu_pool, want_top16, 0, false,
                                 nullptr, 0, nullptr, &l0_served, toks);

        if (pp2) {
            CHECK(cudaEventRecord(ev_s0_done, stages[0].stream));
            // Dispatch Stage 1 kernel enqueue to dedicated core-14 thread WHILE core-15 immediately services Stage 0 cold experts!
            s1_job.W = W;
            s1_job.need_logits = need_logits;
            s1_job.save_snaps = save_snaps;
            s1_job.fast_greedy = fast_greedy;
            s1_job.want_top16 = want_top16;
            const uint32_t jseq = s1_req_seq.load(std::memory_order_relaxed) + 1u;
            s1_req_seq.store(jseq, std::memory_order_release);
            const auto tw1 = std::chrono::steady_clock::now();

            stages[0].service_cold_layers(host_pack, &cpu_pool, s0_base, W, false, l0_served);
            const auto tw2 = std::chrono::steady_clock::now();

            while (s1_done_seq.load(std::memory_order_acquire) != jseq) _mm_pause();
            const uint32_t s1_base = s1_job.s1_base;
            stages[1].service_cold_layers(host_pack, &cpu_pool, s1_base, W);
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamSynchronize(stages[1].stream));
            stages[0].promote_cold(host_pack, W, promote);
            stages[1].promote_cold(host_pack, W, promote);
            const auto tw3 = std::chrono::steady_clock::now();
            if (!is_prefill_step) {
                prof_enq_ms += std::chrono::duration<double, std::milli>(tw1 - tw0).count();
                prof_s0_ms  += std::chrono::duration<double, std::milli>(tw2 - tw1).count();
                prof_s1_ms  += std::chrono::duration<double, std::milli>(tw3 - tw2).count();
                ++prof_wins;
            }
        } else {
            if (need_logits && !fast_greedy && !want_top16) {
                CHECK(cudaMemcpyAsync(h_logits, stages[0].hlogits.p, (size_t) W * NVOCAB * 4,
                                      cudaMemcpyDeviceToHost, stages[0].stream));
            }
            stages[0].service_cold_layers(host_pack, &cpu_pool, s0_base, W);
            stages[0].promote_cold(host_pack, W, promote);
            CHECK(cudaStreamSynchronize(stages[0].stream));
        }
    };

    cudaEvent_t ev_s1_rcopied = nullptr;
    if (pp2) {
        CHECK(cudaSetDevice(stages[1].dev));
        CHECK(cudaEventCreateWithFlags(&ev_s1_rcopied, cudaEventDisableTiming));
    }

    // Stage `[p_start .. p_start + B)` embeddings & 16-head 3-gram PLE rows across 8 threads in ~2 ms!
    auto stage_emb_ple_pf = [&](const int* all_ids, int p_start, int B, float* dst_emb, float* dst_ple) {
        const int prev_m2 = (p_start >= 2) ? all_ids[p_start - 2] : ((p_start == 1) ? ple_prev[1] : ple_prev[0]);
        const int prev_m1 = (p_start >= 1) ? all_ids[p_start - 1] : ple_prev[1];
        auto worker = [&](int tid_w, int t0, int t1) {
            if (tid_w >= 0) {
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(tid_w % 16, &cs);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cs);
            }
            for (int t = t0; t < t1; ++t) {
                stage_one_emb(dst_emb + (size_t) t * NE, all_ids[p_start + t]);
                const int cur = all_ids[p_start + t];
                const int p0 = (t >= 2) ? all_ids[p_start + t - 2] : ((t == 1) ? prev_m1 : prev_m2);
                const int p1 = (t >= 1) ? all_ids[p_start + t - 1] : prev_m1;
                const int32_t t3[3] = {p0, p1, cur};
                int64_t ctx[PLE_NG];
                ctx[0] = cur;
                bool cut = false;
                for (int s = 1; s < PLE_NG; ++s) {
                    const int32_t v = cut ? -1 : t3[2 - s];
                    cut = cut || v < 0 || v == PLE_EOS;
                    ctx[s] = cut ? PLE_EOS : v;
                }
                float* pdst = dst_ple + (size_t) t * NE;
                for (int n = 2; n <= PLE_NG; ++n) {
                    uint64_t mixed = (uint64_t) ctx[0] * ple_mult[0];
                    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * ple_mult[j];
                    for (int gi = 0; gi < PLE_HPN; ++gi) {
                        const int h = (n - 2) * PLE_HPN + gi;
                        const uint32_t r = (uint32_t) (mixed % ple_vocab[h] + ple_off[h]);
                        plesrc.row_direct(r, pdst + (size_t) h * PLE_HEAD);
                    }
                }
            }
        };
        if (B < 128) {
            worker(-1, 0, B);
        } else {
            const int nth = 16;
            std::vector<std::thread> ths;
            const int per = (B + nth - 1) / nth;
            for (int i = 0; i < nth; ++i) {
                const int a = i * per, b = std::min(B, a + per);
                if (a < b) ths.emplace_back(worker, i, a, b);
            }
            for (auto& th : ths) th.join();
        }
        if (B >= 2) {
            ple_prev[0] = all_ids[p_start + B - 2];
            ple_prev[1] = all_ids[p_start + B - 1];
        } else if (B == 1) {
            ple_prev[0] = ple_prev[1];
            ple_prev[1] = all_ids[p_start];
        }
    };

    auto seed_mtp_tail = [&](const int* all_ids, int nprompt, int p_start, int B, const float* R_final_dev) {
        const int tail_floor = std::max(0, nprompt - 1 - MTP_SEED_TAIL);
        if (mtp.enabled && p_start + B > tail_floor) {
            StageGpu& last = stages.back();
            CHECK(cudaSetDevice(last.dev));
            const int sub0 = std::max(0, tail_floor - p_start);
            if (p_start + sub0 == tail_floor && tail_floor > 0) {
                CHECK(cudaMemcpyAsync(mtp.d_kv_floor, &tail_floor, 4, cudaMemcpyHostToDevice, mtp.stream));
            }
            for (int off = sub0; off < B; off += MtpEngine::MAX_B) {
                const int cnt = std::min((int) MtpEngine::MAX_B, B - off);
                *mtp.h_pos = p_start + off;
                for (int j = 0; j < cnt; ++j)
                    stage_one_emb(mtp.h_emb_stage + (size_t) j * NE, all_ids[p_start + off + 1 + j]);
                CHECK(cudaMemcpyAsync(mtp.d_pos, mtp.h_pos, 4, cudaMemcpyHostToDevice, mtp.stream));
                CHECK(cudaMemcpyAsync(mtp.emb.p, mtp.h_emb_stage, (size_t) cnt * NE * 4, cudaMemcpyHostToDevice, mtp.stream));
                const float* src_R = R_final_dev + (size_t) off * HCD;
                CHECK(cudaMemcpyAsync(mtp.Rin.p, src_R, (size_t) cnt * HCD * 4, cudaMemcpyDeviceToDevice, mtp.stream));
                mtp.run_front_kv(last.cos_tab, last.sin_tab, cnt);
                CHECK(cudaStreamSynchronize(mtp.stream));
            }
        }
    };

    // Run prefill over `[p_begin .. p_end)` (`p_end = nprompt - 1`) with Dual-GPU Pipeline Overlap across chunks!
    auto run_prefill_range = [&](const int* all_ids, int nprompt, int p_begin, int p_end) {
        const int total_pf = p_end - p_begin;
        if (total_pf <= 0) return;
        if (total_pf <= 192) {
            for (int p = p_begin; p < p_end; ) {
                const int step_w = std::min((int) MAX_W, p_end - p);
                run_verify_window(all_ids + p, step_w, p, false, true, false, false, true);
                seed_mtp_tail(all_ids, nprompt, p, step_w, stages.back().R.p);
                p += step_w;
            }
            return;
        }

        cpu_pool.set_sleep(true);
        const bool pf_prof = (std::getenv("WZY_PF_PROF") != nullptr);
        double ple_prep_ms = 0.0, mtp_seed_ms = 0.0;

        struct ChunkSpan { int p; int B; };
        std::vector<ChunkSpan> chunks;
        int eff_chunk = prefill_chunk;
        if (pp2 && !explicit_pf_chunk && split_pf_chunks > 1 && total_pf >= 16384) {
            const int want_n = std::max(split_pf_chunks, (total_pf + prefill_chunk - 1) / prefill_chunk);
            eff_chunk = std::min(prefill_chunk, ((total_pf + want_n - 1) / want_n + 63) & ~63);
        }
        for (int p = p_begin; p < p_end; p += eff_chunk)
            chunks.push_back({p, std::min(eff_chunk, p_end - p)});

        std::thread s1_thread;
        std::atomic<bool> s1_p2p_queued{true};
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const int p = chunks[ci].p, B = chunks[ci].B;
            if (ci > 0) {
                CHECK(cudaSetDevice(stages[0].dev));
                CHECK(cudaEventSynchronize(pf_ctx[0].ev_in_copied));
            }
            const auto t_ple0 = std::chrono::steady_clock::now();
            stage_emb_ple_pf(all_ids, p, B, pf_ctx[0].h_emb_pf, pf_ctx[0].h_ple_pf);
            inject_img_emb(pf_ctx[0].h_emb_pf, p, B);
            ple_prep_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ple0).count();

            CHECK(cudaSetDevice(stages[0].dev));
            if (ci > 0 && pp2) {
                CHECK(cudaEventSynchronize(ev_s0_done));
                while (!s1_p2p_queued.load(std::memory_order_acquire)) _mm_pause();
                CHECK(cudaStreamWaitEvent(stages[0].stream, ev_s1_rcopied, 0));
            }
            float* R_s0 = pf_ctx[0].run_stage_prefill(stages[0], p, B, host_pack, nullptr);

            if (!pp2) {
                CHECK(cudaStreamSynchronize(stages[0].stream));
                const auto t_ms0 = std::chrono::steady_clock::now();
                seed_mtp_tail(all_ids, nprompt, p, B, R_s0);
                mtp_seed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ms0).count();
                continue;
            }

            CHECK(cudaEventRecord(ev_s0_done, stages[0].stream));
            if (s1_thread.joinable()) s1_thread.join();
            s1_p2p_queued.store(false, std::memory_order_relaxed);

            s1_thread = std::thread([&, p, B, R_s0]() {
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(14, &cs);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cs);
                CHECK(cudaSetDevice(stages[1].dev));
                CHECK(cudaStreamWaitEvent(stages[1].stream, ev_s0_done, 0));
                float* R_s1 = pf_ctx[1].run_stage_prefill(stages[1], p, B, host_pack, R_s0, ev_s1_rcopied, &s1_p2p_queued);
                CHECK(cudaStreamSynchronize(stages[1].stream));
                const auto t_ms0 = std::chrono::steady_clock::now();
                seed_mtp_tail(all_ids, nprompt, p, B, R_s1);
                mtp_seed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ms0).count();
            });
        }
        if (s1_thread.joinable()) s1_thread.join();
        CHECK(cudaSetDevice(stages[0].dev));
        CHECK(cudaStreamSynchronize(stages[0].stream));
        for (int si = 0; si < nstages; ++si) pf_ctx[si].sync_stats(stages[si]);
        cpu_pool.set_sleep(false);
        if (pf_prof) {
            std::fprintf(stderr, "[PF-HOST total_pf=%d chunks=%zu] ple_prep=%.1fms mtp_seed=%.1fms\n",
                         total_pf, chunks.size(), ple_prep_ms, mtp_seed_ms);
        }
    };

    auto run_prefill_chunk = [&](const int* all_ids, int nprompt, int p_start, int B) {
        run_prefill_range(all_ids, nprompt, p_start, p_start + B);
    };

    // MTP catch-up on `n_cells` newly confirmed cells `[cell0 .. cell0 + n_cells - 1]` (whose pre-head residuals
    // sit in `stages.back().R_pre_head[0 .. n_cells - 1]` and whose confirmed next tokens are `next_toks[0 .. n_cells - 1]`),
    // followed by chaining up to `max_drafts` speculative proposals!
    auto mtp_propose = [&](int cell0, int n_cells, const int* next_toks, int max_drafts,
                           float spec_min_p, const std::vector<int>& stops, std::vector<int>& drafts_out,
                           const int* sfx_cand = nullptr, int sfx_k = 0, int sfx_match = 0,
                           std::vector<int>* b_toks_out = nullptr, SuffixDrafter* sfx_ext = nullptr) {
        drafts_out.clear();
        if (b_toks_out) b_toks_out->clear();
        if (!mtp.enabled || max_drafts <= 0 || n_cells <= 0) return;
        StageGpu& last = stages.back();
        CHECK(cudaSetDevice(last.dev));

        // 1. If n_cells > 1, append KV for cells `0 .. n_cells - 2` first
        if (n_cells > 1) {
            const int kcells = n_cells - 1;
            mtp.h_pos[1] = cell0;
            float* emb_kv = mtp.h_emb_stage + NE;
            for (int j = 0; j < kcells; ++j)
                stage_one_emb(emb_kv + (size_t) j * NE, next_toks[j]);
            CHECK(cudaMemcpyAsync(mtp.d_pos, mtp.h_pos + 1, 4, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaMemcpyAsync(mtp.emb.p, emb_kv, (size_t) kcells * NE * 4, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaMemcpyAsync(mtp.Rin.p, last.R_pre_head.p, (size_t) kcells * HCD * 4, cudaMemcpyDeviceToDevice, mtp.stream));
            mtp.run_front_kv(last.cos_tab, last.sin_tab, kcells);
        }

        // 2. Full MTP step on the last confirmed cell `last_cell = cell0 + n_cells - 1` -> produces `d_1`!
        const int last_cell = cell0 + n_cells - 1;
        *mtp.h_pos = last_cell;
        stage_one_emb(mtp.h_emb_stage, next_toks[n_cells - 1]);
        CHECK(cudaMemcpyAsync(mtp.Rin.p, last.R_pre_head.p + (size_t) (n_cells - 1) * HCD,
                              (size_t) HCD * 4, cudaMemcpyDeviceToDevice, mtp.stream));
        CHECK(cudaGraphLaunch(mtp.step1_exec, mtp.stream));
        int warm_p0 = ple_prev[0], warm_p1 = ple_prev[1];
        warm_ple_token(next_toks[n_cells - 1], warm_p0, warm_p1);
        CHECK(cudaStreamSynchronize(mtp.stream));
        int d_tok = *mtp.h_out_tok;
        float d_prob = *mtp.h_out_prob;
        auto is_stop_id = [&](int id) {
            for (int s : stops) if (id == s) return true;
            return false;
        };
        if (is_stop_id(d_tok)) return;

        // Only take SuffixDrafter when `sfx_match >= 6` AND its first token agrees with the neural MTP drafter `d_tok`!
        if (sfx_cand != nullptr && sfx_k >= 2 && sfx_match >= 6 && sfx_cand[0] == d_tok) {
            const int a_cap = std::min(sfx_match >= 10 ? 7 : 5, max_drafts);
            for (int i = 0; i < std::min(a_cap, sfx_k) && !is_stop_id(sfx_cand[i]); ++i) {
                if (i > 0 && sfx_cand[i] == drafts_out.back()) break;
                drafts_out.push_back(sfx_cand[i]);
                warm_ple_token(sfx_cand[i], warm_p0, warm_p1);
            }
            return;
        }

        // Measured: drafts beyond ~3 are rejected ~70% of the time while every extra row still costs a full
        // forward + MTP chain step, so gate the chain on confidence (WZY_SPEC_DEPTH_{2..7} to retune).
        static const int wzy_spec_depth = std::getenv("WZY_SPEC_DEPTH") ? std::atoi(std::getenv("WZY_SPEC_DEPTH")) : 5;
        const float step0_min_p = std::min(0.24f, spec_min_p);
        if (d_prob < step0_min_p) return;
        if (d_tok == next_toks[n_cells - 1] && d_prob < 0.85f) return;
        drafts_out.push_back(d_tok);
        float cum_prob = d_prob;

        // WZY_SPEC_GATE scales the confidence gates: <1 chains deeper (more rows per window), >1 stops earlier.
        static const float wzy_spec_gate = std::getenv("WZY_SPEC_GATE") ? (float) std::atof(std::getenv("WZY_SPEC_GATE")) : 1.0f;
        const int mtp_cap = std::min(wzy_spec_depth, std::max(1, max_drafts));
        for (int j = 1; j < mtp_cap; ++j) {
            const float explore_p = ((j == 1) ? std::min(0.34f, spec_min_p)
                                  : (j == 2) ? std::min(0.46f, spec_min_p)
                                  : (j == 3) ? 0.65f : 0.78f) * wzy_spec_gate;
            const float min_cum = ((j <= 1) ? 0.0f : (j == 2) ? 0.24f : (j == 3) ? 0.36f : 0.48f) * wzy_spec_gate;
            if (d_prob < explore_p || cum_prob < min_cum) break;
            if (last_cell + j >= ctx_max - 4) break;
            const int prev_d = d_tok;
            *mtp.h_pos = last_cell + j;
            stage_one_emb(mtp.h_emb_stage, d_tok);
            CHECK(cudaGraphLaunch(mtp.step1_exec, mtp.stream));
            warm_ple_token(d_tok, warm_p0, warm_p1);
            CHECK(cudaStreamSynchronize(mtp.stream));
            d_tok = *mtp.h_out_tok;
            d_prob = *mtp.h_out_prob;
            if (is_stop_id(d_tok) || d_tok == prev_d) break;
            if (drafts_out.size() >= 2 && d_tok == drafts_out[drafts_out.size() - 2]) break;
            const float keep_p = ((j == 1) ? std::min(0.36f, spec_min_p)
                               : (j == 2) ? std::min(0.48f, spec_min_p)
                               : (j == 3) ? 0.62f : 0.74f) * wzy_spec_gate;
            if (d_prob < keep_p) break;
            drafts_out.push_back(d_tok);
            cum_prob *= d_prob;
        }
        // Zero-GPU-Cost Hybrid Suffix Tail Extension: if MTP chain landed on a known >=6-gram in context, extend up to max_drafts!
        if (sfx_ext != nullptr && (int) drafts_out.size() >= 2 && (int) drafts_out.size() < max_drafts && d_prob >= 0.58f) {
            int tail_buf[MAX_W];
            const int rem = max_drafts - (int) drafts_out.size();
            const int tk = sfx_ext->propose_after(drafts_out.data(), (int) drafts_out.size(), std::min(rem, 3), tail_buf);
            if (tk > 0 && sfx_ext->last_match() >= 6) {
                for (int i = 0; i < tk && !is_stop_id(tail_buf[i]); ++i) {
                    if (tail_buf[i] == drafts_out.back()) break;
                    drafts_out.push_back(tail_buf[i]);
                    warm_ple_token(tail_buf[i], warm_p0, warm_p1);
                }
            }
        }

        // ---- Dual-window pipeline: extend the SAME chain by one more window (B). B's row-0 input is the token at
        // position `cur_pos + W`, i.e. the model's own next token for window A, so B is only valid when A is fully
        // accepted AND that token equals `b_toks[0]`. The extra steps run only when the caller asks for them.
        if (b_toks_out != nullptr && !drafts_out.empty()) {
            static const int pipe_tok = std::getenv("WZY_PIPE_TOK") ? std::max(2, std::atoi(std::getenv("WZY_PIPE_TOK"))) : 4;
            int cur = d_tok;
            const int base_cell = last_cell + (int) drafts_out.size();
            for (int j = 0; j < pipe_tok; ++j) {
                if (base_cell + j >= ctx_max - 4) break;
                *mtp.h_pos = base_cell + j;
                stage_one_emb(mtp.h_emb_stage, cur);
                CHECK(cudaGraphLaunch(mtp.step1_exec, mtp.stream));
                warm_ple_token(cur, warm_p0, warm_p1);
                CHECK(cudaStreamSynchronize(mtp.stream));
                const int bt = *mtp.h_out_tok;
                if (is_stop_id(bt) || bt == cur) break;
                b_toks_out->push_back(bt);
                cur = bt;
            }
        }
    };
    std::mt19937 rng(std::random_device{}());
    // CLI / Benchmark Mode (WM_TOK & WM_GEN)
    if (const char* tv = std::getenv("WM_TOK")) {
        std::vector<int> prompt;
        int cur = 0; bool any = false;
        for (const char* pch = tv; ; ++pch) {
            if (*pch >= '0' && *pch <= '9') { cur = cur * 10 + (*pch - '0'); any = true; }
            else { if (any) prompt.push_back(cur); cur = 0; any = false; if (*pch == 0) break; }
        }
        const int ngen = std::getenv("WM_GEN") ? std::atoi(std::getenv("WM_GEN")) : ((int) prompt.size() + 16);
        const bool score_on = std::getenv("WM_SCORE") != nullptr;
        GenCfg cfg;
        cfg.temp = 0.0f;
        cfg.stops.clear();
        reset_all();
        std::printf("gen:");

        if (score_on) {
            int hits = 0, ntot = 0, pred = 0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int t = 0; t < ngen && t < ctx_max; ++t) {
                if (t > 0 && t < (int) prompt.size()) {
                    if (pred == prompt[t]) ++hits;
                    ++ntot;
                }
                const int in_tok = (t < (int) prompt.size()) ? prompt[t] : pred;
                run_verify_window(&in_tok, 1, t, true, true);
                verify_or_resample(h_logits, NVOCAB, cfg, prompt, -1, pred, rng);
                if (t + 1 >= (int) prompt.size()) { std::printf(" %d", pred); std::fflush(stdout); }
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("score %d/%d (%.4f)\n", hits, ntot, ntot > 0 ? (double) hits / ntot : 0.0);
            std::printf("\ngen: mean %.2f ms (%.1f tok/s)\n", ms / ngen, 1000.0 * ngen / ms);
            return 0;
        }

        const int np = (int) prompt.size();
        const int max_new = std::max(1, ngen - np);
        const auto t_pf0 = std::chrono::steady_clock::now();
        for (int p = 0; p < np - 1; p += prefill_chunk) {
            const int B = std::min(prefill_chunk, (np - 1) - p);
            run_prefill_chunk(prompt.data(), np, p, B);
        }
        run_verify_window(&prompt[np - 1], 1, np - 1, true, true);
        int cur_tok = 0;
        verify_or_resample(h_logits, NVOCAB, cfg, prompt, -1, cur_tok, rng);
        const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();

        const auto t_dec0 = std::chrono::steady_clock::now();
        std::vector<int> history = prompt;
        int emitted = 0;
        int cur_pos = np;
        int unfed_cell0 = np - 1;
        std::vector<int> unfed_next = {cur_tok};
        long long drafts_offered = 0, drafts_accepted = 0;
        std::vector<int> drafts, win_toks;

        while (emitted < max_new && cur_pos < ctx_max - 8) {
            std::printf(" %d", cur_tok);
            std::fflush(stdout);
            history.push_back(cur_tok);
            ++emitted;
            if (emitted >= max_new) break;

            const int want_k = std::min(cfg.spec_k, max_new - emitted);
            mtp_propose(unfed_cell0, (int) unfed_next.size(), unfed_next.data(),
                        want_k, cfg.spec_min_p, cfg.stops, drafts);
            drafts_offered += (long long) drafts.size();

            win_toks.clear();
            win_toks.push_back(cur_tok);
            for (int d : drafts) win_toks.push_back(d);
            const int W = (int) win_toks.size();

            run_verify_window(win_toks.data(), W, cur_pos, true, true);

            int n_acc = 0;
            int bonus_tok = 0;
            for (int j = 0; j < W; ++j) {
                const int d_cand = (j + 1 < W) ? win_toks[j + 1] : -1;
                int sampled = 0;
                const bool ok = verify_or_resample(h_logits + (size_t) j * NVOCAB, NVOCAB, cfg, history, d_cand, sampled, rng);
                if (j + 1 < W && ok) {
                    ++n_acc;
                    ++drafts_accepted;
                    std::printf(" %d", sampled);
                    std::fflush(stdout);
                    history.push_back(sampled);
                    ++emitted;
                    if (emitted >= max_new) { bonus_tok = sampled; break; }
                } else {
                    bonus_tok = sampled;
                    break;
                }
            }
            const int acc_rows = std::min(W, 1 + n_acc);
            if (W > 1) {
                for (auto& stg : stages) stg.commit_window_state(acc_rows - 1);
                stage_ple_window(win_toks.data(), acc_rows, true);
            }

            unfed_cell0 = cur_pos;
            unfed_next.clear();
            for (int j = 1; j < acc_rows; ++j) unfed_next.push_back(win_toks[j]);
            unfed_next.push_back(bonus_tok);

            cur_pos += acc_rows;
            cur_tok = bonus_tok;
        }
        const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0).count();
        long long lk = 0, vh = 0;
        for (const auto& s : stages) { lk += s.total_lookups; vh += s.total_vram_hits; }
        std::printf("\nprefill: %d tok in %.2f ms (%.1f tok/s) | decode: %d tok in %.2f ms (%.2f tok/s) | accept %lld/%lld (%.1f%%) | vram_hit %.2f%%\n",
                    np, pf_ms, pf_ms > 0 ? 1000.0 * np / pf_ms : 0.0,
                    emitted, dec_ms, dec_ms > 0 ? 1000.0 * emitted / dec_ms : 0.0,
                    drafts_accepted, drafts_offered, drafts_offered > 0 ? 100.0 * drafts_accepted / drafts_offered : 0.0,
                    lk > 0 ? 100.0 * vh / lk : 100.0);
        return 0;
    }

    // Persistent Streaming Serve Mode (JSON-line protocol on stdin/stdout)
    std::printf("READY {\"engine\":\"wzy-pp2-v2-tc-mtp\",\"gpus\":%d,\"split\":%d,\"vram_slots\":%d,\"mtp\":%s,\"ctx_max\":%d,\"boot_s\":%.2f}\n",
                nstages, split_layer, total_vram_slots, mtp.enabled ? "true" : "false", ctx_max, boot_s);
    std::fflush(stdout);

    struct StageSnap {
        std::vector<float> conv_all;
        std::vector<float> ssm_all;
        std::vector<float> ple_hist;
        std::vector<float> idx_tail;
        std::vector<float> idx_dead;
        int saved_idx_blocks = 0;
        std::vector<float> idx_pooled;
        int saved_vram_cells = 0;
        std::vector<signed char> vk_q, vv_q;
        std::vector<unsigned short> vk_s, vv_s;
        int saved_host_cells = 0;
        std::vector<signed char> hk_q, hv_q;
        std::vector<unsigned short> hk_s, hv_s;
    };
    struct ParkedSession {
        std::vector<int> tokens;
        int ple_prev[2] = {-1, -1};
        std::vector<StageSnap> stg;
        bool mtp_valid = false;
        int mtp_kv_floor = 0;
        std::vector<signed char> mtp_vk_q, mtp_vv_q;
        std::vector<unsigned short> mtp_vk_s, mtp_vv_s;
        void clear() { tokens.clear(); stg.clear(); mtp_valid = false; }
    };

    auto save_session_state = [&](ParkedSession& dst, const std::vector<int>& cur_tokens) {
        const int n_tok = (int) cur_tokens.size();
        dst.tokens = cur_tokens;
        dst.ple_prev[0] = ple_prev[0];
        dst.ple_prev[1] = ple_prev[1];
        dst.stg.resize(nstages);
        for (int si = 0; si < nstages; ++si) {
            StageGpu& s = stages[si];
            StageSnap& sn = dst.stg[si];
            CHECK(cudaSetDevice(s.dev));
            sn.conv_all.resize((size_t) s.n_gdn * QKV * 3);
            sn.ssm_all.resize((size_t) s.n_gdn * VH * ST * ST);
            for (int l = s.l_begin; l < s.l_end; ++l) {
                if (!s.L[l].qsa) {
                    const int gl = s.L[l].gdn_local;
                    CHECK(cudaMemcpyAsync(sn.conv_all.data() + (size_t) gl * QKV * 3, s.conv[l].p,
                                          (size_t) QKV * 3 * 4, cudaMemcpyDeviceToHost, s.stream));
                    CHECK(cudaMemcpyAsync(sn.ssm_all.data() + (size_t) gl * VH * ST * ST, s.ssm[l].p,
                                          (size_t) VH * ST * ST * 4, cudaMemcpyDeviceToHost, s.stream));
                }
            }
            if (s.has_ple && s.ple_hist.p) {
                sn.ple_hist.resize((size_t) HCD * PLE_HIST);
                CHECK(cudaMemcpyAsync(sn.ple_hist.data(), s.ple_hist.p,
                                      (size_t) HCD * PLE_HIST * 4, cudaMemcpyDeviceToHost, s.stream));
            }
            if (s.n_qsa > 0) {
                sn.idx_tail.resize(s.idx_tail.n);
                sn.idx_dead.resize(s.idx_dead.n);
                CHECK(cudaMemcpyAsync(sn.idx_tail.data(), s.idx_tail.p, s.idx_tail.n * 4, cudaMemcpyDeviceToHost, s.stream));
                CHECK(cudaMemcpyAsync(sn.idx_dead.data(), s.idx_dead.p, s.idx_dead.n * 4, cudaMemcpyDeviceToHost, s.stream));
                sn.saved_idx_blocks = std::min((int) MAX_IDX_BLOCKS, n_tok / IDX_BLK + 4);
                sn.idx_pooled.resize((size_t) s.n_qsa * sn.saved_idx_blocks * IDX_DIM);
                for (int ql = 0; ql < s.n_qsa; ++ql) {
                    CHECK(cudaMemcpyAsync(sn.idx_pooled.data() + (size_t) ql * sn.saved_idx_blocks * IDX_DIM,
                                          s.idx_pooled.p + (size_t) ql * MAX_IDX_BLOCKS * IDX_DIM,
                                          (size_t) sn.saved_idx_blocks * IDX_DIM * 4, cudaMemcpyDeviceToHost, s.stream));
                }
                sn.saved_vram_cells = std::min((int) KV_VRAM_RING, n_tok);
                sn.vk_q.resize((size_t) s.n_qsa * sn.saved_vram_cells * KVD);
                sn.vv_q.resize((size_t) s.n_qsa * sn.saved_vram_cells * KVD);
                sn.vk_s.resize((size_t) s.n_qsa * sn.saved_vram_cells * KV_SC_PER_CELL);
                sn.vv_s.resize((size_t) s.n_qsa * sn.saved_vram_cells * KV_SC_PER_CELL);
                for (int ql = 0; ql < s.n_qsa; ++ql) {
                    CHECK(cudaMemcpyAsync(sn.vk_q.data() + (size_t) ql * sn.saved_vram_cells * KVD,
                                          s.vk_q + (size_t) ql * KV_VRAM_RING * KVD,
                                          (size_t) sn.saved_vram_cells * KVD, cudaMemcpyDeviceToHost, s.stream));
                    CHECK(cudaMemcpyAsync(sn.vv_q.data() + (size_t) ql * sn.saved_vram_cells * KVD,
                                          s.vv_q + (size_t) ql * KV_VRAM_RING * KVD,
                                          (size_t) sn.saved_vram_cells * KVD, cudaMemcpyDeviceToHost, s.stream));
                    CHECK(cudaMemcpyAsync(sn.vk_s.data() + (size_t) ql * sn.saved_vram_cells * KV_SC_PER_CELL,
                                          s.vk_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL,
                                          (size_t) sn.saved_vram_cells * KV_SC_PER_CELL * 2, cudaMemcpyDeviceToHost, s.stream));
                    CHECK(cudaMemcpyAsync(sn.vv_s.data() + (size_t) ql * sn.saved_vram_cells * KV_SC_PER_CELL,
                                          s.vv_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL,
                                          (size_t) sn.saved_vram_cells * KV_SC_PER_CELL * 2, cudaMemcpyDeviceToHost, s.stream));
                }
                if (s.hk_q != nullptr && n_tok > KV_VRAM_RING && n_tok <= 98304) {
                    sn.saved_host_cells = n_tok;
                    sn.hk_q.resize((size_t) s.n_qsa * sn.saved_host_cells * KVD);
                    sn.hv_q.resize((size_t) s.n_qsa * sn.saved_host_cells * KVD);
                    sn.hk_s.resize((size_t) s.n_qsa * sn.saved_host_cells * KV_SC_PER_CELL);
                    sn.hv_s.resize((size_t) s.n_qsa * sn.saved_host_cells * KV_SC_PER_CELL);
                    for (int ql = 0; ql < s.n_qsa; ++ql) {
                        std::memcpy(sn.hk_q.data() + (size_t) ql * sn.saved_host_cells * KVD,
                                    s.hk_q + (size_t) ql * s.ctx_max * KVD, (size_t) sn.saved_host_cells * KVD);
                        std::memcpy(sn.hv_q.data() + (size_t) ql * sn.saved_host_cells * KVD,
                                    s.hv_q + (size_t) ql * s.ctx_max * KVD, (size_t) sn.saved_host_cells * KVD);
                        std::memcpy(sn.hk_s.data() + (size_t) ql * sn.saved_host_cells * KV_SC_PER_CELL,
                                    s.hk_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL, (size_t) sn.saved_host_cells * KV_SC_PER_CELL * 2);
                        std::memcpy(sn.hv_s.data() + (size_t) ql * sn.saved_host_cells * KV_SC_PER_CELL,
                                    s.hv_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL, (size_t) sn.saved_host_cells * KV_SC_PER_CELL * 2);
                    }
                } else {
                    sn.saved_host_cells = 0;
                }
            }
        }
        if (mtp.enabled) {
            dst.mtp_valid = true;
            dst.mtp_vk_q.resize((size_t) MTP_KV_RING * KVD);
            dst.mtp_vv_q.resize((size_t) MTP_KV_RING * KVD);
            dst.mtp_vk_s.resize((size_t) MTP_KV_RING * KV_SC_PER_CELL);
            dst.mtp_vv_s.resize((size_t) MTP_KV_RING * KV_SC_PER_CELL);
            CHECK(cudaSetDevice(mtp.dev));
            CHECK(cudaMemcpyAsync(dst.mtp_vk_q.data(), mtp.vk_q, (size_t) MTP_KV_RING * KVD, cudaMemcpyDeviceToHost, mtp.stream));
            CHECK(cudaMemcpyAsync(dst.mtp_vv_q.data(), mtp.vv_q, (size_t) MTP_KV_RING * KVD, cudaMemcpyDeviceToHost, mtp.stream));
            CHECK(cudaMemcpyAsync(dst.mtp_vk_s.data(), mtp.vk_s, (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2, cudaMemcpyDeviceToHost, mtp.stream));
            CHECK(cudaMemcpyAsync(dst.mtp_vv_s.data(), mtp.vv_s, (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2, cudaMemcpyDeviceToHost, mtp.stream));
            CHECK(cudaMemcpyAsync(&dst.mtp_kv_floor, mtp.d_kv_floor, 4, cudaMemcpyDeviceToHost, mtp.stream));
            CHECK(cudaStreamSynchronize(mtp.stream));
        }
        for (int si = 0; si < nstages; ++si) {
            CHECK(cudaSetDevice(stages[si].dev));
            CHECK(cudaStreamSynchronize(stages[si].stream));
        }
    };

    auto restore_session_state = [&](const ParkedSession& src, std::vector<int>& cur_tokens) {
        cur_tokens = src.tokens;
        ple_prev[0] = src.ple_prev[0];
        ple_prev[1] = src.ple_prev[1];
        for (int si = 0; si < nstages; ++si) {
            StageGpu& s = stages[si];
            const StageSnap& sn = src.stg[si];
            CHECK(cudaSetDevice(s.dev));
            for (int l = s.l_begin; l < s.l_end; ++l) {
                if (!s.L[l].qsa) {
                    const int gl = s.L[l].gdn_local;
                    CHECK(cudaMemcpyAsync(s.conv[l].p, sn.conv_all.data() + (size_t) gl * QKV * 3,
                                          (size_t) QKV * 3 * 4, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(s.ssm[l].p, sn.ssm_all.data() + (size_t) gl * VH * ST * ST,
                                          (size_t) VH * ST * ST * 4, cudaMemcpyHostToDevice, s.stream));
                }
            }
            if (s.has_ple && s.ple_hist.p && !sn.ple_hist.empty()) {
                CHECK(cudaMemcpyAsync(s.ple_hist.p, sn.ple_hist.data(),
                                      (size_t) HCD * PLE_HIST * 4, cudaMemcpyHostToDevice, s.stream));
            }
            if (s.n_qsa > 0) {
                CHECK(cudaMemcpyAsync(s.idx_tail.p, sn.idx_tail.data(), s.idx_tail.n * 4, cudaMemcpyHostToDevice, s.stream));
                CHECK(cudaMemcpyAsync(s.idx_dead.p, sn.idx_dead.data(), s.idx_dead.n * 4, cudaMemcpyHostToDevice, s.stream));
                for (int ql = 0; ql < s.n_qsa; ++ql) {
                    CHECK(cudaMemcpyAsync(s.idx_pooled.p + (size_t) ql * MAX_IDX_BLOCKS * IDX_DIM,
                                          sn.idx_pooled.data() + (size_t) ql * sn.saved_idx_blocks * IDX_DIM,
                                          (size_t) sn.saved_idx_blocks * IDX_DIM * 4, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(s.vk_q + (size_t) ql * KV_VRAM_RING * KVD,
                                          sn.vk_q.data() + (size_t) ql * sn.saved_vram_cells * KVD,
                                          (size_t) sn.saved_vram_cells * KVD, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(s.vv_q + (size_t) ql * KV_VRAM_RING * KVD,
                                          sn.vv_q.data() + (size_t) ql * sn.saved_vram_cells * KVD,
                                          (size_t) sn.saved_vram_cells * KVD, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(s.vk_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL,
                                          sn.vk_s.data() + (size_t) ql * sn.saved_vram_cells * KV_SC_PER_CELL,
                                          (size_t) sn.saved_vram_cells * KV_SC_PER_CELL * 2, cudaMemcpyHostToDevice, s.stream));
                    CHECK(cudaMemcpyAsync(s.vv_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL,
                                          sn.vv_s.data() + (size_t) ql * sn.saved_vram_cells * KV_SC_PER_CELL,
                                          (size_t) sn.saved_vram_cells * KV_SC_PER_CELL * 2, cudaMemcpyHostToDevice, s.stream));
                }
                if (s.hk_q != nullptr) {
                    if (sn.saved_host_cells > 0) {
                        for (int ql = 0; ql < s.n_qsa; ++ql) {
                            std::memcpy(s.hk_q + (size_t) ql * s.ctx_max * KVD,
                                        sn.hk_q.data() + (size_t) ql * sn.saved_host_cells * KVD, (size_t) sn.saved_host_cells * KVD);
                            std::memcpy(s.hv_q + (size_t) ql * s.ctx_max * KVD,
                                        sn.hv_q.data() + (size_t) ql * sn.saved_host_cells * KVD, (size_t) sn.saved_host_cells * KVD);
                            std::memcpy(s.hk_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL,
                                        sn.hk_s.data() + (size_t) ql * sn.saved_host_cells * KV_SC_PER_CELL, (size_t) sn.saved_host_cells * KV_SC_PER_CELL * 2);
                            std::memcpy(s.hv_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL,
                                        sn.hv_s.data() + (size_t) ql * sn.saved_host_cells * KV_SC_PER_CELL, (size_t) sn.saved_host_cells * KV_SC_PER_CELL * 2);
                        }
                    } else if (sn.saved_vram_cells > 0) {
                        // For n_tok <= KV_VRAM_RING, kv_vram_slot(p) == p, so sn.vk_q IS the exact [0 .. n_tok) host mirror!
                        for (int ql = 0; ql < s.n_qsa; ++ql) {
                            std::memcpy(s.hk_q + (size_t) ql * s.ctx_max * KVD,
                                        sn.vk_q.data() + (size_t) ql * sn.saved_vram_cells * KVD, (size_t) sn.saved_vram_cells * KVD);
                            std::memcpy(s.hv_q + (size_t) ql * s.ctx_max * KVD,
                                        sn.vv_q.data() + (size_t) ql * sn.saved_vram_cells * KVD, (size_t) sn.saved_vram_cells * KVD);
                            std::memcpy(s.hk_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL,
                                        sn.vk_s.data() + (size_t) ql * sn.saved_vram_cells * KV_SC_PER_CELL, (size_t) sn.saved_vram_cells * KV_SC_PER_CELL * 2);
                            std::memcpy(s.hv_s + (size_t) ql * s.ctx_max * KV_SC_PER_CELL,
                                        sn.vv_s.data() + (size_t) ql * sn.saved_vram_cells * KV_SC_PER_CELL, (size_t) sn.saved_vram_cells * KV_SC_PER_CELL * 2);
                        }
                    }
                }
            }
        }
        if (mtp.enabled && src.mtp_valid) {
            CHECK(cudaSetDevice(mtp.dev));
            CHECK(cudaMemcpyAsync(mtp.vk_q, src.mtp_vk_q.data(), (size_t) MTP_KV_RING * KVD, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaMemcpyAsync(mtp.vv_q, src.mtp_vv_q.data(), (size_t) MTP_KV_RING * KVD, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaMemcpyAsync(mtp.vk_s, src.mtp_vk_s.data(), (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaMemcpyAsync(mtp.vv_s, src.mtp_vv_s.data(), (size_t) MTP_KV_RING * KV_SC_PER_CELL * 2, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaMemcpyAsync(mtp.d_kv_floor, &src.mtp_kv_floor, 4, cudaMemcpyHostToDevice, mtp.stream));
            CHECK(cudaStreamSynchronize(mtp.stream));
        }
        for (int si = 0; si < nstages; ++si) {
            CHECK(cudaSetDevice(stages[si].dev));
            CHECK(cudaStreamSynchronize(stages[si].stream));
        }
    };

    std::vector<int> state_tokens;
    ParkedSession parked_session, tmp_session;
    std::signal(SIGUSR1, on_sigusr1_cancel);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "PING") { std::printf("PONG\n"); std::fflush(stdout); continue; }
        if (line == "RESET") { reset_all(); state_tokens.clear(); parked_session.clear(); std::printf("OK\n"); std::fflush(stdout); continue; }
        auto save_learned_profile = [&]() {
            std::vector<int> prior_pos(48 * NEXP, 1000000);
            for (size_t r = 0; r < prior_ranked.size(); ++r) {
                const int gk = prior_ranked[r].first * NEXP + prior_ranked[r].second;
                if (gk >= 0 && gk < 48 * NEXP && prior_pos[gk] == 1000000) prior_pos[gk] = (int) r;
            }
            std::vector<int> tier(48 * NEXP, 3);
            std::vector<uint32_t> freq(48 * NEXP, 0);
            for (int si = 0; si < nstages; ++si) {
                const StageGpu& stg = stages[si];
                const int core_keep = std::min(3200, stg.static_cap * 2 / 3);
                for (int gk = stg.l_begin * NEXP; gk < stg.l_end * NEXP; ++gk) {
                    const int sl = stg.h_lut[gk];
                    if (sl >= 0 && sl < core_keep) tier[gk] = 0;      // Top 3,200 universal/English/Code backbone experts per GPU
                    else if (sl >= stg.static_cap) tier[gk] = 1;      // Pin learned CJK + Agent experts into static_cap [3200..static_cap)!
                    else if (sl >= 0) tier[gk] = 2;                   // Remaining prior experts fill adapt_cap
                    else tier[gk] = 3;
                    freq[gk] = stg.cold_freq[gk];
                }
            }
            std::vector<int> order(48 * NEXP);
            for (int gk = 0; gk < 48 * NEXP; ++gk) order[gk] = gk;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                if (tier[a] != tier[b]) return tier[a] < tier[b];
                if (tier[a] == 3 && freq[a] != freq[b]) return freq[a] > freq[b];
                return prior_pos[a] < prior_pos[b];
            });
            std::ofstream out("/home/liaoyang/wzy-flash-serve/expert-profile-cjk.bin.tmp", std::ios::binary);
            if (out) {
                const char magic[4] = {'S', 'T', 'R', 'P'};
                const uint32_t hdr[5] = {1u, 48u, 512u, 24576u, 24576u};
                out.write(magic, 4);
                out.write((const char*) hdr, 20);
                for (int gk : order) {
                    const uint16_t pr[2] = {(uint16_t) (gk / NEXP), (uint16_t) (gk % NEXP)};
                    out.write((const char*) pr, 4);
                }
                out.close();
                std::rename("/home/liaoyang/wzy-flash-serve/expert-profile-cjk.bin.tmp",
                            "/home/liaoyang/wzy-flash-serve/expert-profile-cjk.bin");
            }
        };
        if (line == "SAVE_PROFILE") {
            save_learned_profile();
            std::printf("OK\n");
            std::fflush(stdout);
            continue;
        }
        if (line == "QUIT" || line == "EXIT") break;

        g_cancel_req = 0;
        std::vector<int> ids;
        GenCfg cfg;
        const bool json_mode = (line[0] == '{');
        if (json_mode) {
            parse_json_ints(line, "\"ids\"", ids);
            std::vector<int> custom_stops;
            parse_json_ints(line, "\"stop\"", custom_stops);
            if (!custom_stops.empty()) cfg.stops = custom_stops;
            cfg.max_new = (int) parse_json_num(line, "\"max_tokens\"", 512);
            cfg.temp = (float) parse_json_num(line, "\"temperature\"", 0.3);
            cfg.top_p = (float) parse_json_num(line, "\"top_p\"", 0.75);
            cfg.top_k = (int) parse_json_num(line, "\"top_k\"", 15);
            cfg.min_p = (float) parse_json_num(line, "\"min_p\"", 0.02);
            cfg.rep_pen = (float) parse_json_num(line, "\"repetition_penalty\"", 1.0);
            cfg.spec_k = (int) parse_json_num(line, "\"spec\"", 7);
            cfg.spec_min_p = (float) parse_json_num(line, "\"spec_min_p\"", 0.72);
            // Optional vision payload: `emb_file` (SVE1 file from `strata-vision`), one run per image:
            // `emb_starts` / `emb_counts` (arrays, prompt order).  `emb_start` / `emb_count` still work for one image.
            // A text-only request clears whatever the previous one loaded.
            std::string emb_file;
            if (parse_json_str(line, "\"emb_file\"", emb_file)) {
                std::vector<int> emb_starts, emb_counts;
                parse_json_ints(line, "\"emb_starts\"", emb_starts);
                parse_json_ints(line, "\"emb_counts\"", emb_counts);
                if (emb_starts.empty()) {
                    emb_starts.push_back((int) parse_json_num(line, "\"emb_start\"", 0));
                    emb_counts.push_back((int) parse_json_num(line, "\"emb_count\"", 0));
                }
                load_img_emb(emb_file, emb_starts, emb_counts);
            } else {
                load_img_emb(std::string(), {}, {});
            }
            cfg.no_reuse = ((int) parse_json_num(line, "\"no_reuse\"", 0) != 0);
        } else {
            const size_t sp = line.rfind(' ');
            if (sp == std::string::npos) continue;
            int total_want = std::atoi(line.c_str() + sp + 1);
            int cv = 0; bool any = false;
            for (size_t i = 0; i < sp; ++i) {
                char ch = line[i];
                if (ch >= '0' && ch <= '9') { cv = cv * 10 + (ch - '0'); any = true; }
                else { if (any) ids.push_back(cv); cv = 0; any = false; }
            }
            if (any) ids.push_back(cv);
            cfg.max_new = std::max(1, total_want - (int) ids.size());
            cfg.temp = 0.0f;
            cfg.stops.clear();
        }

        if (ids.empty()) {
            if (json_mode) std::printf("DONE {\"generated\":0,\"finish\":\"stop\"}\n");
            else std::printf("output :\nEND\n");
            std::fflush(stdout);
            continue;
        }
        if ((int) ids.size() >= ctx_max - 16) {
            ids.erase(ids.begin(), ids.end() - (ctx_max - 128));
        }

        // 2-Slot Longest Common Prefix Reuse (Active GPU Slot + Parked Host Slot for Side-Request / Dual-Conv Protection!)
        int reused = 0;
        bool restored_park = false;
        if (!cfg.no_reuse && !state_tokens.empty()) {
            const size_t lim = std::min(state_tokens.size(), ids.size() - 1);
            size_t m = 0;
            while (m < lim && state_tokens[m] == ids[m]) ++m;
            if (m == state_tokens.size()) reused = (int) m;
        }
        if (reused == 0 && !cfg.no_reuse && !parked_session.tokens.empty()) {
            const size_t lim = std::min(parked_session.tokens.size(), ids.size() - 1);
            size_t m = 0;
            while (m < lim && parked_session.tokens[m] == ids[m]) ++m;
            if (m == parked_session.tokens.size()) {
                // Incoming request matches the parked session! Swap active <-> parked if active is worth keeping
                if (state_tokens.size() >= 64 && state_tokens.size() <= 98304) {
                    save_session_state(tmp_session, state_tokens);
                    restore_session_state(parked_session, state_tokens);
                    parked_session = std::move(tmp_session);
                } else {
                    restore_session_state(parked_session, state_tokens);
                }
                reused = (int) state_tokens.size();
                restored_park = true;
            }
        }
        if (reused == 0) {
            // Before wiping the active session for a new/side request, park it if it's substantial (>= 64 tokens)
            // and don't let a tiny side request (< parked / 2) overwrite a large parked main session!
            if (!cfg.no_reuse && state_tokens.size() >= 64 && state_tokens.size() <= 98304 &&
                (parked_session.tokens.empty() || state_tokens.size() * 2 >= parked_session.tokens.size())) {
                save_session_state(parked_session, state_tokens);
            }
            reset_all();
            state_tokens.clear();
        }

        const int nprompt = (int) ids.size();
        const auto t_pf0 = std::chrono::steady_clock::now();
        const bool had_borrow_possible = (nprompt - 1 - reused > 192);
        run_prefill_range(ids.data(), nprompt, reused, nprompt - 1);
        for (int p = reused; p < nprompt - 1; ++p) state_tokens.push_back(ids[p]);
        if (had_borrow_possible) {
            for (int si = 0; si < nstages; ++si) pf_ctx[si].restore_borrowed(stages[si], host_pack);
            for (int si = 0; si < nstages; ++si) { CHECK(cudaSetDevice(stages[si].dev)); CHECK(cudaStreamSynchronize(stages[si].stream)); }
        }
        const bool no_rep = (std::fabs(cfg.rep_pen - 1.0f) < REP_PEN_EPS);
        const bool fast_greedy = no_rep && (cfg.temp <= 1e-5f);
        // Only use GPU top-16 fast path when repetition penalty is inactive; otherwise a token outside
        // raw top-16 can enter top-k after higher-ranked tokens are penalized.
        const bool want_top16 = no_rep && !fast_greedy && (cfg.top_k > 0 && cfg.top_k <= 16);
        run_verify_window(&ids[nprompt - 1], 1, nprompt - 1, true, true, fast_greedy, want_top16);
        state_tokens.push_back(ids[nprompt - 1]);

        std::vector<int> history = ids;
        int cur_tok = 0;
        if (fast_greedy) {
            cur_tok = stages.back().h_greedy_ids[0];
        } else if (want_top16) {
            verify_or_resample_top16(stages.back().h_top16_val, stages.back().h_top16_id, cfg, history, (size_t) nprompt, -1, cur_tok, rng);
        } else {
            verify_or_resample(h_logits, NVOCAB, cfg, history, -1, cur_tok, rng);
        }

        const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
        const int pf_read = nprompt - reused;
        if (json_mode) {
            std::printf("PREFILL {\"prompt_tokens\":%d,\"reused\":%d,\"read\":%d,\"prompt_ms\":%.2f,\"prompt_tps\":%.1f,\"restored_park\":%d,\"parked_tokens\":%d}\n",
                        nprompt, reused, pf_read, pf_ms, pf_ms > 0 ? 1000.0 * pf_read / pf_ms : 0.0,
                        restored_park ? 1 : 0, (int) parked_session.tokens.size());
            std::fflush(stdout);
        } else {
            std::printf("output :");
        }
        long long lk0 = 0, vh0 = 0, pr0 = 0;
        for (const auto& s : stages) { lk0 += s.total_lookups; vh0 += s.total_vram_hits; pr0 += s.total_promotions; }
        prof_enq_ms = prof_s0_ms = prof_s1_ms = prof_mtp_ms = 0.0;
        prof_wins = 0;
        const auto t_dec0 = std::chrono::steady_clock::now();
        static const int wzy_pipe_tok = std::getenv("WZY_PIPE_TOK") ? std::max(0, std::atoi(std::getenv("WZY_PIPE_TOK"))) : 0;
        int generated = 0;
        const char* finish = "length";
        int cur_pos = nprompt;
        int unfed_cell0 = nprompt - 1;
        std::vector<int> unfed_next = {cur_tok};
        long long drafts_offered = 0, drafts_accepted = 0;
        std::vector<int> drafts, win_toks, b_toks;
        SuffixDrafter sfx(3, 32, std::max<size_t>(65536, ids.size() + (size_t) cfg.max_new + 1024));
        sfx.append(ids.data(), ids.size());
        int sfx_buf[MAX_W + 4];

        auto is_stop_tok = [&](int t) {
            for (int s : cfg.stops) if (t == s) return true;
            return false;
        };

        auto verify_one_row = [&](int j, int d_cand, int& sampled) -> bool {
            if (fast_greedy) {
                sampled = stages.back().h_greedy_ids[j];
                return (d_cand >= 0 && d_cand == sampled);
            } else if (want_top16) {
                return verify_or_resample_top16(stages.back().h_top16_val + j * 16,
                                                stages.back().h_top16_id + j * 16,
                                                cfg, history, (size_t) nprompt, d_cand, sampled, rng);
            } else {
                return verify_or_resample(h_logits + (size_t) j * NVOCAB, NVOCAB, cfg, history, d_cand, sampled, rng);
            }
        };

        while (generated < cfg.max_new && cur_pos < ctx_max - 8) {
            if (g_cancel_req) { finish = "cancelled"; break; }
            if (is_stop_tok(cur_tok)) { finish = "stop"; break; }
            ++generated;
            history.push_back(cur_tok);
            sfx.append_one(cur_tok);
            if (json_mode) { std::printf("TOK %d\n", cur_tok); std::fflush(stdout); }
            else { std::printf(" %d", cur_tok); std::fflush(stdout); }

            if (generated >= cfg.max_new || cur_pos >= ctx_max - 8) { finish = "length"; break; }

            // With the pipeline on, A's snapshots occupy [0, W) and B's [W, W+W_B), so both must fit in MAX_W.
            const int depth_cap = wzy_pipe_tok > 0 ? std::max(1, MAX_W / 2 - 1) : (MAX_W - 1);
            const int want_k = std::min(std::min(cfg.spec_k, depth_cap), cfg.max_new - generated);
            const int sfx_k = sfx.propose(MAX_W - 1, sfx_buf);
            const int sfx_match = sfx.last_match();
            const auto tm0 = std::chrono::steady_clock::now();
            b_toks.clear();
            mtp_propose(unfed_cell0, (int) unfed_next.size(), unfed_next.data(),
                        want_k, cfg.spec_min_p, cfg.stops, drafts,
                        sfx_buf, sfx_k, sfx_match, wzy_pipe_tok > 0 ? &b_toks : nullptr, &sfx);
            prof_mtp_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm0).count();

            drafts_offered += (long long) drafts.size();
            win_toks.clear();
            win_toks.push_back(cur_tok);
            for (int d : drafts) win_toks.push_back(d);
            const int W = (int) win_toks.size();
            // Window B is only usable when its speculative tokens fit the verify budget and the window limit.
            const int b_cap = std::min(MAX_W - W, wzy_pipe_tok);
            const int W_B = (wzy_pipe_tok > 0 && b_cap >= 2 && (int) b_toks.size() >= 2 &&
                             generated + 1 + std::min((int) b_toks.size(), b_cap) <= cfg.max_new)
                                ? std::min((int) b_toks.size(), b_cap)
                                : 0;
            if (W_B <= 0) {
                run_verify_window(win_toks.data(), W, cur_pos, true, true, fast_greedy, want_top16);

                int n_acc = 0, bonus_tok = 0;
                bool hit_stop = false;
                for (int j = 0; j < W; ++j) {
                    const int d_cand = (j + 1 < W) ? win_toks[j + 1] : -1;
                    int sampled = 0;
                    const bool ok = verify_one_row(j, d_cand, sampled);
                    if (j + 1 < W && ok) {
                        if (is_stop_tok(sampled)) { hit_stop = true; finish = "stop"; break; }
                        ++n_acc; ++drafts_accepted; ++generated;
                        history.push_back(sampled); sfx.append_one(sampled);
                        if (json_mode) { std::printf("TOK %d\n", sampled); std::fflush(stdout); }
                        else { std::printf(" %d", sampled); std::fflush(stdout); }
                        if (generated >= cfg.max_new) { bonus_tok = sampled; break; }
                    } else {
                        bonus_tok = sampled;
                        if (is_stop_tok(bonus_tok)) { hit_stop = true; finish = "stop"; }
                        break;
                    }
                }
                const int acc_rows = std::min(W, 1 + n_acc);
                if (W > 1) {
                    // When `acc_rows == W` (all drafts accepted!), GDN/PLE/QSA states are already committed in-place by the kernels!
                    if (acc_rows < W) {
                        for (auto& stg : stages) stg.commit_window_state(acc_rows - 1, cur_pos + acc_rows - 1);
                    }
                    advance_ple_prev(win_toks.data(), acc_rows);
                }
                for (int j = 0; j < acc_rows; ++j) state_tokens.push_back(win_toks[j]);
                if (hit_stop || generated >= cfg.max_new || g_cancel_req) {
                    if (g_cancel_req && !hit_stop) finish = "cancelled";
                    break;
                }
                unfed_cell0 = cur_pos;
                unfed_next.clear();
                for (int j = 1; j < acc_rows; ++j) unfed_next.push_back(win_toks[j]);
                unfed_next.push_back(bonus_tok);
                cur_pos += acc_rows;
                cur_tok = bonus_tok;
                continue;
            }

            // ---- DUAL-GPU OVERLAPPED PIPELINE (`--pipeline-windows 2`): GPU1 verifies Window A || GPU0 runs Window B! ----
            h_pos[0] = cur_pos;
            for (int t = 0; t < W; ++t) stage_one_emb(h_emb + (size_t) t * NE, win_toks[t]);
            stage_ple_window(win_toks.data(), W, false, h_ple);

            h_pos[1] = cur_pos + W;
            float* h_emb_b = h_emb + (size_t) MAX_W * NE;
            float* h_ple_b = h_ple + (size_t) MAX_W * NE;
            for (int t = 0; t < W_B; ++t) stage_one_emb(h_emb_b + (size_t) t * NE, b_toks[t]);
            const int b_prev[2] = {
                W >= 2 ? win_toks[W - 2] : ple_prev[1],
                win_toks[W - 1]
            };
            stage_ple_window(b_toks.data(), W_B, false, h_ple_b, b_prev);

            CHECK(cudaSetDevice(stages[0].dev));
            const uint32_t s0_a_base = stages[0].cold_seq;
            stages[0].record_forward(h_pos, h_emb, h_ple, W, false, true, host_pack, &cpu_pool, want_top16, 0, false);
            CHECK(cudaMemcpyAsync(stages[0].R_handoff.p, stages[0].R.p, (size_t) W * HCD * 4, cudaMemcpyDeviceToDevice, stages[0].stream));
            CHECK(cudaEventRecord(ev_s0_done, stages[0].stream));

            // 2. Enqueue Window A on Stage 1 (waiting on ev_s0_done)
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamWaitEvent(stages[1].stream, ev_s0_done, 0));
            CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev, stages[0].R_handoff.p, stages[0].dev, (size_t) W * HCD * 4, stages[1].stream));
            const uint32_t s1_a_base = stages[1].cold_seq;
            stages[1].record_forward(h_pos, h_emb, h_ple, W, true, W > 1, host_pack, &cpu_pool, want_top16, 0, false);
            if (!fast_greedy && !want_top16) {
                CHECK(cudaMemcpyAsync(h_logits, stages[1].hlogits.p, (size_t) W * NVOCAB * 4, cudaMemcpyDeviceToHost, stages[1].stream));
            }

            // 3. Service Stage 0 for A, then IMMEDIATELY launch speculative Window B on Stage 0 while Stage 1 computes A!
            stages[0].service_cold_layers(host_pack, &cpu_pool, s0_a_base, W);
            stages[0].promote_cold(host_pack, W, true);
            stages[0].commit_window_state(W - 1, cur_pos + W - 1);
            const uint32_t s0_b_base = stages[0].cold_seq;
            const int b_snap = W;   // B's per-step snapshots start right after A's [0, W)
            stages[0].record_forward(h_pos + 1, h_emb_b, h_ple_b, W_B, false, true, host_pack, &cpu_pool, want_top16, b_snap, false);
            int l0_b = stages[0].l_begin, l1_a = stages[1].l_begin;
            while (l1_a < stages[1].l_end) {
                const uint32_t want1 = s1_a_base + (uint32_t) (l1_a - stages[1].l_begin) + 1u;
                if (stages[1].h_cold_req[l1_a].ready_seq == want1) {
                    stages[1].service_one_cold_layer(l1_a, host_pack, &cpu_pool, s1_a_base, W, false);
                    ++l1_a;
                    continue;
                }
                if (l0_b < stages[0].l_end) {
                    const uint32_t want0 = s0_b_base + (uint32_t) (l0_b - stages[0].l_begin) + 1u;
                    if (stages[0].h_cold_req[l0_b].ready_seq == want0) {
                        stages[0].service_one_cold_layer(l0_b, host_pack, &cpu_pool, s0_b_base, W_B, false);
                        ++l0_b;
                        continue;
                    }
                }
                _mm_pause();
            }
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamSynchronize(stages[1].stream));
            stages[1].promote_cold(host_pack, W, true);
            int n_acc = 0, bonus_tok = 0;
            bool hit_stop = false;
            for (int j = 0; j < W; ++j) {
                const int d_cand = (j + 1 < W) ? win_toks[j + 1] : -1;
                int sampled = 0;
                const bool ok = verify_one_row(j, d_cand, sampled);
                if (j + 1 < W && ok) {
                    if (is_stop_tok(sampled)) { hit_stop = true; finish = "stop"; break; }
                    ++n_acc; ++drafts_accepted; ++generated;
                    history.push_back(sampled); sfx.append_one(sampled);
                    if (json_mode) { std::printf("TOK %d\n", sampled); std::fflush(stdout); }
                    else { std::printf(" %d", sampled); std::fflush(stdout); }
                    if (generated >= cfg.max_new) { bonus_tok = sampled; break; }
                } else {
                    bonus_tok = sampled;
                    if (is_stop_tok(bonus_tok)) { hit_stop = true; finish = "stop"; }
                    break;
                }
            }
            const int acc_rows = std::min(W, 1 + n_acc);
            advance_ple_prev(win_toks.data(), acc_rows);
            for (int j = 0; j < acc_rows; ++j) state_tokens.push_back(win_toks[j]);

            const bool b_hit = (!hit_stop && generated < cfg.max_new && !g_cancel_req && acc_rows == W && bonus_tok == b_toks[0]);
            if (!b_hit) {
                // Speculative B missed: set hardware skip flag (`*d_skip_flag = 1`) & release all remaining layers in 1 us!
                set_wzy_abort(stages[0].abort_stream, 1);
                CHECK(cudaStreamSynchronize(stages[0].abort_stream));
                _mm_sfence();
                std::atomic_thread_fence(std::memory_order_seq_cst);
                for (int l = l0_b; l < stages[0].l_end; ++l) {
                    stages[0].h_cold_req[l].cpu_done_seq = s0_b_base + (uint32_t) (l - stages[0].l_begin) + 1u;
                }
                _mm_sfence();
                std::atomic_thread_fence(std::memory_order_seq_cst);
                stages[0].pending_promo.clear();
                CHECK(cudaSetDevice(stages[0].dev));
                CHECK(cudaStreamSynchronize(stages[0].stream));
                set_wzy_abort(stages[0].stream, 0);
                set_wzy_abort(stages[0].abort_stream, 0);
                CHECK(cudaStreamSynchronize(stages[0].abort_stream));
                if (acc_rows < W) stages[0].commit_window_state(acc_rows - 1, cur_pos + acc_rows - 1);
                else stages[0].commit_window_state(W - 1, cur_pos + W - 1);
                if (W > 1) stages[1].commit_window_state(acc_rows - 1, cur_pos + acc_rows - 1);
                if (hit_stop || generated >= cfg.max_new || g_cancel_req) {
                    if (g_cancel_req && !hit_stop) finish = "cancelled";
                    break;
                }
                unfed_cell0 = cur_pos;
                unfed_next.clear();
                for (int j = 1; j < acc_rows; ++j) unfed_next.push_back(win_toks[j]);
                unfed_next.push_back(bonus_tok);
                cur_pos += acc_rows;
                cur_tok = bonus_tok;
                continue;
            }

            // ---- PIPELINE HIT (`acc_rows == W && bonus_tok == b_toks[0]`): finish any remaining layers `[l0_b .. l_end)` of B! ----
            stages[0].service_cold_layers(host_pack, &cpu_pool, s0_b_base, W_B, false, l0_b);
            if (W > 1) stages[1].commit_window_state(W - 1, cur_pos + W - 1);
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaMemcpyAsync(stages[1].R_handoff.p, stages[1].R_pre_head.p, (size_t) W * HCD * 4, cudaMemcpyDeviceToDevice, stages[1].stream));

            // Emit `b_toks[0]` (`== bonus_tok` of A!)
            ++generated;
            history.push_back(b_toks[0]);
            sfx.append_one(b_toks[0]);
            if (json_mode) { std::printf("TOK %d\n", b_toks[0]); std::fflush(stdout); }
            else { std::printf(" %d", b_toks[0]); std::fflush(stdout); }

            if (generated >= cfg.max_new || cur_pos + W >= ctx_max - 8) {
                CHECK(cudaSetDevice(stages[0].dev));
                CHECK(cudaStreamSynchronize(stages[0].stream));
                stages[0].commit_window_state(W - 1, cur_pos + W - 1);
                state_tokens.push_back(b_toks[0]);
                finish = "length";
                break;
            }

            const int W_B_eff = std::min(W_B, 1 + (cfg.max_new - generated));
            drafts_offered += (long long) (W_B_eff - 1);

            CHECK(cudaSetDevice(stages[0].dev));
            CHECK(cudaEventRecord(ev_s0_done, stages[0].stream));
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamWaitEvent(stages[1].stream, ev_s0_done, 0));
            CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev, stages[0].R.p, stages[0].dev, (size_t) W_B_eff * HCD * 4, stages[1].stream));
            const uint32_t s1_b_base = stages[1].cold_seq;
            stages[1].record_forward(h_pos + 1, h_emb_b, h_ple_b, W_B_eff, true, W_B_eff > 1, host_pack, &cpu_pool, want_top16, b_snap);
            if (!fast_greedy && !want_top16) {
                CHECK(cudaMemcpyAsync(h_logits, stages[1].hlogits.p, (size_t) W_B_eff * NVOCAB * 4, cudaMemcpyDeviceToHost, stages[1].stream));
            }
            stages[0].promote_cold(host_pack, W_B_eff, true);
            stages[1].service_cold_layers(host_pack, &cpu_pool, s1_b_base, W_B_eff);
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamSynchronize(stages[1].stream));
            stages[1].promote_cold(host_pack, W_B_eff, true);

            int n_acc_b = 0, bonus_tok_b = 0;
            bool hit_stop_b = false;
            for (int j = 0; j < W_B_eff; ++j) {
                const int d_cand = (j + 1 < W_B_eff) ? b_toks[j + 1] : -1;
                int sampled = 0;
                const bool ok = verify_one_row(j, d_cand, sampled);
                if (j + 1 < W_B_eff && ok) {
                    if (is_stop_tok(sampled)) { hit_stop_b = true; finish = "stop"; break; }
                    ++n_acc_b; ++drafts_accepted; ++generated;
                    history.push_back(sampled); sfx.append_one(sampled);
                    if (json_mode) { std::printf("TOK %d\n", sampled); std::fflush(stdout); }
                    else { std::printf(" %d", sampled); std::fflush(stdout); }
                    if (generated >= cfg.max_new) { bonus_tok_b = sampled; break; }
                } else {
                    bonus_tok_b = sampled;
                    if (is_stop_tok(bonus_tok_b)) { hit_stop_b = true; finish = "stop"; }
                    break;
                }
            }
            const int acc_rows_b = std::min(W_B_eff, 1 + n_acc_b);
            stages[0].commit_window_state(b_snap + acc_rows_b - 1, cur_pos + W + acc_rows_b - 1);
            if (W_B_eff > 1) stages[1].commit_window_state(b_snap + acc_rows_b - 1, cur_pos + W + acc_rows_b - 1);
            stage_ple_window(b_toks.data(), acc_rows_b, true);
            for (int j = 0; j < acc_rows_b; ++j) state_tokens.push_back(b_toks[j]);

            if (hit_stop_b || generated >= cfg.max_new || g_cancel_req) {
                if (g_cancel_req && !hit_stop_b) finish = "cancelled";
                break;
            }

            // Concatenate A's `W` pre-head residuals (`R_handoff`) + B's `acc_rows_b` pre-head residuals into `R_pre_head`!
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaMemcpyAsync(stages[1].R.p, stages[1].R_pre_head.p, (size_t) acc_rows_b * HCD * 4, cudaMemcpyDeviceToDevice, stages[1].stream));
            CHECK(cudaMemcpyAsync(stages[1].R_pre_head.p, stages[1].R_handoff.p, (size_t) W * HCD * 4, cudaMemcpyDeviceToDevice, stages[1].stream));
            CHECK(cudaMemcpyAsync(stages[1].R_pre_head.p + (size_t) W * HCD, stages[1].R.p, (size_t) acc_rows_b * HCD * 4, cudaMemcpyDeviceToDevice, stages[1].stream));

            unfed_cell0 = cur_pos;
            unfed_next.clear();
            for (int j = 1; j < W; ++j) unfed_next.push_back(win_toks[j]);
            for (int j = 0; j < acc_rows_b; ++j) unfed_next.push_back(b_toks[j]);
            unfed_next.push_back(bonus_tok_b);

            cur_pos += W + acc_rows_b;
            cur_tok = bonus_tok_b;
        }

        const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0).count();
        long long lk1 = 0, vh1 = 0, pr1 = 0;
        for (const auto& s : stages) { lk1 += s.total_lookups; vh1 += s.total_vram_hits; pr1 += s.total_promotions; }
        const long long dlk = lk1 - lk0, dvh = vh1 - vh0, dpr = pr1 - pr0;
        const double hit_rate = dlk > 0 ? (double) dvh / (double) dlk : 1.0;
        if (prof_wins > 0 && std::getenv("WZY_DEC_PROF") != nullptr) {
            std::fprintf(stderr, "[DEC-PROF gen=%d wins=%d (%.2f tok/win)] total=%.1fms (%.2f ms/win) | enq=%.2fms s0=%.2fms s1=%.2fms mtp=%.2fms (per win: enq=%.2f s0=%.2f s1=%.2f mtp=%.2f) | hit=%.2f%% promo=%lld\n",
                         generated, prof_wins, (double) generated / prof_wins, dec_ms, dec_ms / prof_wins,
                         prof_enq_ms, prof_s0_ms, prof_s1_ms, prof_mtp_ms,
                         prof_enq_ms / prof_wins, prof_s0_ms / prof_wins, prof_s1_ms / prof_wins, prof_mtp_ms / prof_wins,
                         hit_rate * 100.0, dpr);
        }
        if (json_mode) {
            std::printf("DONE {\"generated\":%d,\"prompt_tokens\":%d,\"reused\":%d,\"prompt_ms\":%.2f,\"decode_ms\":%.2f,\"decode_tps\":%.2f,\"draft_n\":%lld,\"draft_accepted\":%lld,\"vram_hit\":%.4f,\"promoted\":%lld,\"restored_park\":%d,\"parked_tokens\":%d,\"active_tokens\":%d,\"finish\":\"%s\"}\n",
                        generated, nprompt, reused, pf_ms, dec_ms,
                        dec_ms > 0 ? 1000.0 * generated / dec_ms : 0.0,
                        drafts_offered, drafts_accepted, hit_rate, dpr,
                        restored_park ? 1 : 0, (int) parked_session.tokens.size(), (int) state_tokens.size(), finish);
            std::fflush(stdout);
        } else {
            std::printf("\nEND\n");
            std::fflush(stdout);
        }
        // Restore any borrowed prefill tail slots while idle between requests!
        for (int si = 0; si < nstages; ++si) pf_ctx[si].restore_borrowed(stages[si], host_pack);
        for (int si = 0; si < nstages; ++si) { CHECK(cudaSetDevice(stages[si].dev)); CHECK(cudaStreamSynchronize(stages[si].stream)); }
        // Profile is only saved on explicit SAVE_PROFILE command so single-language requests never overwrite the bilingual CJK+Agent profile.
    }
    s1_stop.store(true, std::memory_order_release);
    if (s1_enq_worker.joinable()) s1_enq_worker.join();
    return 0;
}
