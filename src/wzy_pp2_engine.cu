// wzy_pp2_engine.cu -- Main Driver, PP2 Loader, Graph Capture, Sampler & Streaming Protocol
#include "wzy_stage.cuh"
#include <iostream>

static bool is_qsa_layer(int l) { return (l % 4) == 3; }

struct GenCfg {
    int max_new = 512;
    float temp = 0.0f;
    float top_p = 0.75f;
    int top_k = 20;
    float min_p = 0.02f;
    float rep_pen = 1.0f;
    bool no_reuse = false;
    std::vector<int> stops = {248046, 248044}; // <|im_end|>, <|endoftext|>
};

static int sample_logits(const float* logits, int nvocab, const GenCfg& cfg,
                         const std::vector<int>& history, std::mt19937& rng) {
    if (cfg.temp <= 1e-5f && std::fabs(cfg.rep_pen - 1.0f) < 1e-5f) {
        int best = 0;
        for (int v = 1; v < nvocab; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    }
    static std::vector<float> buf;
    buf.assign(logits, logits + nvocab);
    if (std::fabs(cfg.rep_pen - 1.0f) >= 1e-5f && !history.empty()) {
        const size_t start = history.size() > 512 ? history.size() - 512 : 0;
        static std::vector<uint8_t> seen;
        seen.assign(nvocab, 0);
        for (size_t i = start; i < history.size(); ++i) {
            const int id = history[i];
            if (id >= 0 && id < nvocab && !seen[id]) {
                seen[id] = 1;
                buf[id] = (buf[id] > 0.f) ? (buf[id] / cfg.rep_pen) : (buf[id] * cfg.rep_pen);
            }
        }
    }
    if (cfg.temp <= 1e-5f) {
        int best = 0;
        for (int v = 1; v < nvocab; ++v) if (buf[v] > buf[best]) best = v;
        return best;
    }
    const int k = (cfg.top_k > 0 && cfg.top_k < nvocab) ? cfg.top_k : 64;
    static std::vector<int> idx;
    idx.resize(nvocab);
    for (int i = 0; i < nvocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return buf[a] > buf[b]; });
    const float inv_t = 1.0f / cfg.temp;
    const float mx = buf[idx[0]] * inv_t;
    std::vector<float> probs(k);
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(buf[idx[i]] * inv_t - mx);
        sum += probs[i];
    }
    for (int i = 0; i < k; ++i) probs[i] /= sum;
    const float p_max = probs[0];
    int keep = k;
    float cum = 0.f;
    for (int i = 0; i < k; ++i) {
        if (i > 0 && cfg.min_p > 0.f && probs[i] < cfg.min_p * p_max) {
            keep = i;
            break;
        }
        cum += probs[i];
        if (cfg.top_p > 0.f && cfg.top_p < 1.0f && cum >= cfg.top_p) {
            keep = i + 1;
            break;
        }
    }
    float psum = 0.f;
    for (int i = 0; i < keep; ++i) psum += probs[i];
    std::uniform_real_distribution<float> dist(0.f, psum);
    float r = dist(rng);
    for (int i = 0; i < keep; ++i) {
        r -= probs[i];
        if (r <= 0.f) return idx[i];
    }
    return idx[keep - 1];
}

static void init_stage(StageGpu& s, int dev, int lb, int le, int ctx_max, const Gguf& g) {
    s.dev = dev;
    s.l_begin = lb;
    s.l_end = le;
    s.ctx_max = ctx_max;
    s.has_ple = (lb <= 1 && 1 < le);
    s.has_head = (le == 48);
    CHECK(cudaSetDevice(dev));
    fill_tables_current_device();
    CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));

    s.L.resize(48);
    s.conv.resize(48);
    s.ssm.resize(48);
    s.n_qsa = 0;
    for (int l = lb; l < le; ++l) {
        LayerW& w = s.L[l];
        w.qsa = is_qsa_layer(l);
        if (w.qsa) w.qsa_local = s.n_qsa++;
        char name[256];
        auto q = [&](const char* suffix) { std::snprintf(name, sizeof name, "blk.%d.%s", l, suffix); return name; };
        for (int h = 0; h < 2; ++h) {
            const char* pre = h == 0 ? "hc_attn_" : "hc_ffn_";
            std::snprintf(name, sizeof name, "blk.%d.%snorm.weight", l, pre);
            w.hc_norm[h] = up_f32(g, name);
            std::snprintf(name, sizeof name, "blk.%d.%sdown.weight", l, pre);
            w.hc_down[h] = up_bf(g, name);
            std::snprintf(name, sizeof name, "blk.%d.%sup.weight", l, pre);
            w.hc_up[h] = up_bf(g, name);
            std::snprintf(name, sizeof name, "blk.%d.%sinject.weight", l, pre);
            w.hc_inj[h] = up_bf(g, name);
        }
        if (!w.qsa) {
            w.qkv = up_q8(g, q("attn_qkv.weight"));
            w.wgate = up_q8(g, q("attn_gate.weight"));
            w.wout = up_q8(g, q("ssm_out.weight"));
            w.alpha = up_f32(g, q("ssm_alpha.weight"));
            w.beta = up_f32(g, q("ssm_beta.weight"));
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
        }
        w.g_inp = up_f32(g, q("ffn_gate_inp.weight"));
        w.s_gate_inp = up_f32(g, q("ffn_gate_inp_shexp.weight"));
        w.sh_gate = up_q8(g, q("ffn_gate_shexp.weight"));
        w.sh_up = up_q8(g, q("ffn_up_shexp.weight"));
        w.sh_down = up_q8(g, q("ffn_down_shexp.weight"));
    }

    s.kcache = alloc0((size_t) s.n_qsa * ctx_max * KVD);
    s.vcache = alloc0((size_t) s.n_qsa * ctx_max * KVD);
    s.qsa_sc = alloc0((size_t) QH * ctx_max);
    s.R = alloc0(HCD);

    // Scratch
    s.xn = alloc0(HCD); s.lo = alloc0(HCLR); s.gg = alloc0(HCD); s.gated = alloc0(HCD);
    s.mixed = alloc0(NE); s.mixed2 = alloc0(NE); s.inject = alloc0(HC); s.block = alloc0(NE);
    s.qkv = alloc0(QKV); s.convout = alloc0(QKV); s.z = alloc0(GV);
    s.alpha = alloc0(VH); s.beta = alloc0(VH); s.decay = alloc0(VH); s.bt = alloc0(VH);
    s.o = alloc0(GV); s.on = alloc0(GV);
    s.qg = alloc0(2 * QD); s.kk = alloc0(KVD); s.vv = alloc0(KVD); s.attn = alloc0(QD);
    s.slog = alloc0(NEXP); s.sids = alloc0(NEXP); s.swts = alloc0(NEXP);
    s.sff = alloc0(NFF); s.sup = alloc0(NFF); s.sshared = alloc0(NE); s.sgate = alloc0(1); s.tmp_ff = alloc0(NFF);
    s.ff3 = alloc0(TOPK * NFF * 3); s.parts = alloc0(TOPK * NE); s.comb = alloc0(NE);
    s.ffact = alloc0(TOPK * NFF); s.exsc = alloc0(TOPK * 3);

    const int AMAX = 8192;
    CHECK(cudaMalloc(&s.xq8, AMAX));
    CHECK(cudaMalloc(&s.xs8, (AMAX / 32 + 8) * 4));
    CHECK(cudaMalloc(&s.xq4, AMAX));
    CHECK(cudaMalloc(&s.xs4, (AMAX / 16 + 8) * 4));
    CHECK(cudaMalloc(&s.eq32, NE));
    CHECK(cudaMalloc(&s.es32, (size_t) (NE / 32) * 4));
    CHECK(cudaMalloc(&s.eq16, NE));
    CHECK(cudaMalloc(&s.es16, (size_t) (NE / 16) * 4));
    CHECK(cudaMalloc(&s.dq32, (size_t) TOPK * NFF));
    CHECK(cudaMalloc(&s.ds32, (size_t) TOPK * NFF / 32 * 4));
    CHECK(cudaMalloc(&s.dq16, (size_t) TOPK * NFF));
    CHECK(cudaMalloc(&s.ds16, (size_t) TOPK * NFF / 16 * 4));

    if (s.has_ple) {
        s.ple_wk = up_bf(g, "blk.1.ple_key.weight");
        s.ple_wv = up_bf(g, "blk.1.ple_value.weight");
        s.ple_nk = up_f32(g, "blk.1.ple_norm_key.weight");
        s.ple_nq = up_f32(g, "blk.1.ple_norm_query.weight");
        s.ple_nc = up_f32(g, "blk.1.ple_norm_conv.weight");
        s.ple_cv = up_f32(g, "blk.1.ple_conv1d.weight");
        s.ple_hist = alloc0((size_t) HCD * PLE_HIST);
        s.d_ple_emb = alloc0(NE);
        s.ple_q = alloc0(HCD); s.ple_k = alloc0(HCD); s.ple_v = alloc0(NE);
        s.ple_g = alloc0(HC); s.ple_gat = alloc0(HCD); s.ple_n = alloc0(HCD);
    }

    if (s.has_head) {
        s.headnorm = up_f32(g, "output_hc_norm.weight");
        s.head_down = up_bf(g, "output_hc_down.weight");
        s.head_up = up_bf(g, "output_hc_up.weight");
        s.headq = up_q8(g, "output.weight");
        s.hlogits = alloc0(NVOCAB);
    }

    CHECK(cudaMalloc(&s.dctx, 4));
    CHECK(cudaMalloc(&s.dpos, 4));
    CHECK(cudaMalloc(&s.dmisses, 4));
    CHECK(cudaMalloc(&s.dslots, TOPK * 4));
    CHECK(cudaMalloc(&s.mkey, 512 * 4));
    s.h_mkey.resize(512);
    CHECK(cudaMalloc(&s.d_lut, 48 * NEXP * 4));
    s.h_lut.assign(48 * NEXP, -1);
    s.miss_cnt.assign(48 * NEXP, 0);

    // Allocate all remaining VRAM (minus reserve) for this stage's expert cache!
    const size_t reserve_mib = std::getenv("WZY_RESERVE_MIB") ? std::atol(std::getenv("WZY_RESERVE_MIB")) : 650;
    size_t free_b = 0, total_b = 0;
    CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t res_b = reserve_mib * 1024ull * 1024ull;
    const size_t budget = free_b > res_b + 64 * PER ? (free_b - res_b) : (free_b / 2);
    int max_stage_experts = (le - lb) * NEXP;
    s.cap = std::min((int) (budget / PER), max_stage_experts);
    s.adapt_cap = std::min(2500, s.cap * 2 / 5);
    if (std::getenv("WZY_ADAPT_SLOTS")) s.adapt_cap = std::min(s.cap / 2, std::max(0, std::atoi(std::getenv("WZY_ADAPT_SLOTS"))));
    s.static_cap = s.cap - s.adapt_cap;
    s.ring_owner.assign(std::max(1, s.adapt_cap), -1);
    CHECK(cudaMalloc(&s.xblobs, (size_t) s.cap * PER));
    CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::fprintf(stderr, "[wzy-pp2] GPU%d layers [%d..%d): %d VRAM expert slots (%d static + %d adaptive, %.2f GiB), %.0f MiB VRAM free\n",
                 dev, lb, le, s.cap, s.static_cap, s.adapt_cap, (double) s.cap * PER / 1073741824.0, free_b / 1048576.0);
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

int main() {
    const auto t_boot0 = std::chrono::steady_clock::now();
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    const bool pp2 = (ndev >= 2) && (std::getenv("WZY_SINGLE") == nullptr);
    const int split_layer = pp2 ? (std::getenv("WZY_SPLIT") ? std::atoi(std::getenv("WZY_SPLIT")) : 25) : 48;
    const int ctx_max = std::getenv("WZY_CTX") ? std::atoi(std::getenv("WZY_CTX")) : 16384;

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

    const int nstages = pp2 ? 2 : 1;
    std::vector<StageGpu> stages(nstages);
    if (pp2) {
        init_stage(stages[0], 0, 0, split_layer, ctx_max, g);
        init_stage(stages[1], 1, split_layer, 48, ctx_max, g);
    } else {
        init_stage(stages[0], 0, 0, 48, ctx_max, g);
    }

    // Map experts.bin and register in 1 GiB chunks as UVA pinned mapped memory
    const std::string exp_path = std::string(KD) + "/pack/experts.bin";
    const int efd = ::open(exp_path.c_str(), O_RDONLY);
    if (efd < 0) { std::fprintf(stderr, "cannot open %s\n", exp_path.c_str()); return 2; }
    struct stat est{};
    fstat(efd, &est);
    const size_t exp_sz = (size_t) est.st_size;
    uint8_t* host_pack = (uint8_t*) mmap(nullptr, exp_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE, efd, 0);
    ::close(efd);
    if (host_pack == MAP_FAILED) { std::fprintf(stderr, "mmap experts.bin failed\n"); return 2; }

    // Fault in pages in parallel across 16 threads & register for UVA zero-copy PCIe access
    const bool pin_pack = (std::getenv("WZY_NOPIN") == nullptr);
    if (pin_pack) {
        std::fprintf(stderr, "[wzy-pp2] Warming & pinning %.2f GiB expert pack across 16 threads...\n", exp_sz / 1073741824.0);
        const int nth = 16;
        std::vector<std::thread> workers;
        for (int tid = 0; tid < nth; ++tid) {
            workers.emplace_back([&, tid]() {
                const size_t pages = exp_sz / 4096;
                volatile uint8_t sink = 0;
                for (size_t pg = tid; pg < pages; pg += nth) {
                    sink ^= host_pack[pg * 4096];
                }
                (void) sink;
            });
        }
        for (auto& th : workers) th.join();

        const size_t CHUNK = (size_t) 256 * PER; // 707,792,896 B = lcm(4096, PER)
        size_t pinned_ok = 0;
        if (cudaHostRegister(host_pack, exp_sz, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess) {
            pinned_ok = exp_sz;
        } else {
            cudaGetLastError();
            for (size_t off = 0; off < exp_sz; off += CHUNK) {
                const size_t len = std::min(CHUNK, exp_sz - off);
                if (cudaHostRegister(host_pack + off, len, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess) {
                    pinned_ok += len;
                } else {
                    cudaGetLastError();
                    break;
                }
            }
        }
        std::fprintf(stderr, "[wzy-pp2] Pinned %.2f / %.2f GiB of expert pack for zero-copy UVA\n",
                     pinned_ok / 1073741824.0, exp_sz / 1073741824.0);
    } else {
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

    // Prefill both GPUs' static VRAM slots from expert-profile.bin!
    std::string prof_path = std::getenv("WZY_PROFILE")
        ? std::getenv("WZY_PROFILE")
        : "/home/liaoyang/layftllm/strata-fork/data/expert-profile.bin";
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
        // Fallback uniform fill if profile is absent
        for (int l = 0; l < 48; ++l)
            for (int e = 0; e < NEXP; ++e)
                ranked.emplace_back(l, e);

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
                if (vslot >= stg.static_cap) {
                    stg.ring_owner[vslot - stg.static_cap] = gkey;
                }
                CHECK(cudaMemcpyAsync(stg.xblobs + (size_t) vslot * PER,
                                      host_pack + (size_t) gkey * PER,
                                      PER, cudaMemcpyHostToDevice, stg.stream));
            }
            CHECK(cudaMemcpyAsync(stg.d_lut, stg.h_lut.data(), 48 * NEXP * 4, cudaMemcpyHostToDevice, stg.stream));
            CHECK(cudaStreamSynchronize(stg.stream));
            std::fprintf(stderr, "[wzy-pp2] GPU%d prefilled %d VRAM expert slots (%d static + %d adaptive)\n",
                         stg.dev, filled, stg.static_cap, filled - stg.static_cap);
        }
    }

    // Token embedding (BF16 table preferred, Q8_0 fallback)
    const T* te = g.find("token_embd.weight");
    const uint8_t* embbase = g.at(te);
    const int embnb = NE / 32;
    const uint16_t* bf_emb_base = nullptr;
    {
        const int bfd = ::open((std::string(KD) + "/token-embd-bf16.gguf").c_str(), O_RDONLY);
        if (bfd >= 0) {
            struct stat bst{};
            if (fstat(bfd, &bst) == 0 && (size_t) bst.st_size >= 256 + (size_t) NVOCAB * NE * 2) {
                uint8_t* em = (uint8_t*) mmap(nullptr, (size_t) bst.st_size, PROT_READ, MAP_PRIVATE, bfd, 0);
                if (em != MAP_FAILED && std::memcmp(em, "GGUF", 4) == 0) bf_emb_base = (const uint16_t*) (em + 256);
            }
            ::close(bfd);
        }
    }

    CHECK(cudaSetDevice(0));
    float* h_emb = nullptr;
    float* h_ple = nullptr;
    int *h_ctx = nullptr, *h_pos = nullptr;
    float* h_logits = nullptr;
    CHECK(cudaHostAlloc(&h_emb, NE * 4, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_ple, (size_t) HCD * 4, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_ctx, 4, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_pos, 4, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_logits, (size_t) NVOCAB * 4, cudaHostAllocPortable));

    auto set_emb = [&](int id) {
        if (id < 0 || id >= NVOCAB) id = 0;
        if (bf_emb_base != nullptr) {
            const uint16_t* rbf = bf_emb_base + (size_t) id * NE;
            for (int i = 0; i < NE; ++i) h_emb[i] = bf2f(rbf[i]);
        } else {
            const uint8_t* row = embbase + (size_t) id * embnb * 34;
            for (int b = 0; b < embnb; ++b) {
                const float d = h2f(*(const uint16_t*) (row + b * 34));
                const int8_t* qp = (const int8_t*) (row + b * 34 + 2);
                for (int i = 0; i < 32; ++i) h_emb[b * 32 + i] = d * qp[i];
            }
        }
    };

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
    auto ple_gather = [&](int cur, bool advance) {
        const int32_t t3[3] = {ple_prev[0], ple_prev[1], cur};
        int64_t ctx[PLE_NG];
        ctx[0] = cur;
        bool cut = false;
        for (int s = 1; s < PLE_NG; ++s) {
            const int32_t t = cut ? -1 : t3[2 - s];
            cut = cut || t < 0 || t == PLE_EOS;
            ctx[s] = cut ? PLE_EOS : t;
        }
        for (int n = 2; n <= PLE_NG; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * ple_mult[j];
            for (int gi = 0; gi < PLE_HPN; ++gi) {
                const int h = (n - 2) * PLE_HPN + gi;
                const uint32_t r = (uint32_t) (mixed % ple_vocab[h] + ple_off[h]);
                plesrc.row(r, h_ple + (size_t) h * PLE_HEAD);
            }
        }
        if (advance) { ple_prev[0] = ple_prev[1]; ple_prev[1] = cur; }
    };

    auto reset_all = [&]() {
        for (auto& stg : stages) stg.reset_state();
        ple_prev[0] = -1;
        ple_prev[1] = -1;
    };

    // Warm up & capture independent CUDA graphs per GPU stage (exec_body without head, exec_full with head)
    *h_ctx = 0; *h_pos = 0;
    set_emb(760);
    ple_gather(760, false);
    for (int si = 0; si < nstages; ++si) {
        StageGpu& stg = stages[si];
        CHECK(cudaSetDevice(stg.dev));
        stg.record_forward(h_ctx, h_pos, h_emb, h_ple, true);
        CHECK(cudaStreamSynchronize(stg.stream));
        stg.reset_state();

        // Capture body graph (skips 676MB output head during prompt prefill!)
        cudaGraph_t gb = nullptr;
        CHECK(cudaStreamBeginCapture(stg.stream, cudaStreamCaptureModeThreadLocal));
        stg.record_forward(h_ctx, h_pos, h_emb, h_ple, false);
        CHECK(cudaStreamEndCapture(stg.stream, &gb));
        CHECK(cudaGraphInstantiate(&stg.exec_body, gb, nullptr, nullptr, 0));

        if (stg.has_head) {
            cudaGraph_t gf = nullptr;
            CHECK(cudaStreamBeginCapture(stg.stream, cudaStreamCaptureModeThreadLocal));
            stg.record_forward(h_ctx, h_pos, h_emb, h_ple, true);
            CHECK(cudaStreamEndCapture(stg.stream, &gf));
            CHECK(cudaGraphInstantiate(&stg.exec_full, gf, nullptr, nullptr, 0));
        } else {
            stg.exec_full = stg.exec_body;
        }
    }
    reset_all();
    for (auto& stg : stages) { stg.total_lookups = 0; stg.total_vram_hits = 0; stg.total_promotions = 0; }

    const double boot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_boot0).count();
    int total_vram_slots = 0;
    for (const auto& stg : stages) total_vram_slots += stg.cap;
    std::fprintf(stderr, "[wzy-pp2] Ready in %.1f s | GPUs: %d | split: %d | total VRAM slots: %d / 24576 (%.1f%%) | ctx_max: %d\n",
                 boot_s, nstages, split_layer, total_vram_slots, 100.0 * total_vram_slots / 24576.0, ctx_max);

    cudaEvent_t ev_s0_done = nullptr;
    if (pp2) {
        CHECK(cudaSetDevice(stages[0].dev));
        CHECK(cudaEventCreateWithFlags(&ev_s0_done, cudaEventDisableTiming));
    }

    // Execute one token across Stage 0 -> NVLink -> Stage 1 with overlapped GPU0 promotion & GPU1 compute
    auto step_token = [&](int tok, int pos, bool need_logits, bool promote) {
        *h_ctx = pos;
        *h_pos = pos;
        set_emb(tok);
        ple_gather(tok, true);

        // 1. Launch Stage 0 on GPU 0
        CHECK(cudaSetDevice(stages[0].dev));
        cudaGraphExec_t g0 = (!pp2 && need_logits) ? stages[0].exec_full : stages[0].exec_body;
        CHECK(cudaGraphLaunch(g0, stages[0].stream));

        if (pp2) {
            // Record GPU0 completion event & immediately queue NVLink handoff + Stage 1 on GPU 1 without blocking CPU!
            CHECK(cudaEventRecord(ev_s0_done, stages[0].stream));
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamWaitEvent(stages[1].stream, ev_s0_done, 0));
            CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev,
                                      stages[0].R.p, stages[0].dev,
                                      (size_t) HCD * 4, stages[1].stream));
            cudaGraphExec_t g1 = need_logits ? stages[1].exec_full : stages[1].exec_body;
            CHECK(cudaGraphLaunch(g1, stages[1].stream));

            // While GPU 1 executes layers 25..47 + Head, run GPU 0's cold promotion in parallel!
            stages[0].promote_cold(host_pack, promote);

            if (need_logits) {
                CHECK(cudaSetDevice(stages[1].dev));
                CHECK(cudaMemcpyAsync(h_logits, stages[1].hlogits.p, (size_t) NVOCAB * 4,
                                      cudaMemcpyDeviceToHost, stages[1].stream));
            }
            stages[1].promote_cold(host_pack, promote);
        } else {
            if (need_logits) {
                CHECK(cudaMemcpyAsync(h_logits, stages[0].hlogits.p, (size_t) NVOCAB * 4,
                                      cudaMemcpyDeviceToHost, stages[0].stream));
            }
            stages[0].promote_cold(host_pack, promote);
        }
    };

    std::mt19937 rng(std::random_device{}());

    // CLI / Benchmark compatibility mode (WM_TOK & WM_GEN)
    if (const char* tv = std::getenv("WM_TOK")) {
        std::vector<int> prompt;
        int cur = 0; bool any = false;
        for (const char* pch = tv; ; ++pch) {
            if (*pch >= '0' && *pch <= '9') { cur = cur * 10 + (*pch - '0'); any = true; }
            else { if (any) prompt.push_back(cur); cur = 0; any = false; if (*pch == 0) break; }
        }
        const int ngen = std::getenv("WM_GEN") ? std::atoi(std::getenv("WM_GEN")) : ((int) prompt.size() + 8);
        const bool score_on = std::getenv("WM_SCORE") != nullptr;
        GenCfg cfg;
        cfg.temp = 0.0f;
        reset_all();
        std::printf("gen:");
        int hits = 0, ntot = 0, pred = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < ngen && t < ctx_max; ++t) {
            if (score_on && t > 0 && t < (int) prompt.size()) {
                if (pred == prompt[t]) ++hits;
                ++ntot;
            }
            const int in_tok = (t < (int) prompt.size()) ? prompt[t] : pred;
            const bool want_log = score_on || (t + 1 >= (int) prompt.size());
            step_token(in_tok, t, want_log, true);
            if (want_log) pred = sample_logits(h_logits, NVOCAB, cfg, prompt, rng);
            if (t + 1 >= (int) prompt.size()) {
                std::printf(" %d", pred);
                std::fflush(stdout);
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (score_on) std::printf("score %d/%d (%.4f)\n", hits, ntot, ntot > 0 ? (double) hits / ntot : 0.0);
        long long lk = 0, vh = 0;
        for (const auto& s : stages) { lk += s.total_lookups; vh += s.total_vram_hits; }
        std::printf("\ngen: mean %.2f ms (%.1f tok/s), vram_hit %.2f%%\n",
                    ms / ngen, 1000.0 * ngen / ms, lk > 0 ? 100.0 * vh / lk : 0.0);
        return 0;
    }

    // Persistent Streaming Serve Mode (JSON-line protocol on stdin/stdout)
    std::printf("READY {\"engine\":\"wzy-pp2-sm75\",\"gpus\":%d,\"split\":%d,\"vram_slots\":%d,\"ctx_max\":%d,\"boot_s\":%.2f}\n",
                nstages, split_layer, total_vram_slots, ctx_max, boot_s);
    std::fflush(stdout);

    // Track exact token sequence currently sitting in GDN + QSA + PLE recurrent state for prefix reuse!
    std::vector<int> state_tokens;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "PING") {
            std::printf("PONG\n");
            std::fflush(stdout);
            continue;
        }
        if (line == "RESET") {
            reset_all();
            state_tokens.clear();
            std::printf("OK\n");
            std::fflush(stdout);
            continue;
        }
        if (line == "QUIT" || line == "EXIT") break;

        std::vector<int> ids;
        GenCfg cfg;
        bool json_mode = (line[0] == '{');
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
            cfg.no_reuse = (parse_json_num(line, "\"no_reuse\"", 0) != 0);
        } else {
            // Legacy "<id,id,...> <total_len>" compatibility
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
        if ((int) ids.size() >= ctx_max - 4) {
            ids.erase(ids.begin(), ids.end() - (ctx_max - 64));
        }

        // Prefix KV + GDN + PLE continuation check:
        // We must process at least the last prompt token `ids.back()` with `need_logits = true` to produce the first output token!
        int reused = 0;
        if (!cfg.no_reuse && !state_tokens.empty() && state_tokens.size() < ids.size()) {
            bool match = true;
            for (size_t i = 0; i < state_tokens.size(); ++i) {
                if (state_tokens[i] != ids[i]) { match = false; break; }
            }
            if (match) reused = (int) state_tokens.size();
        }
        if (reused == 0) {
            reset_all();
            state_tokens.clear();
        }

        long long lk0 = 0, vh0 = 0, pr0 = 0;
        for (const auto& s : stages) { lk0 += s.total_lookups; vh0 += s.total_vram_hits; pr0 += s.total_promotions; }

        const auto t_pf0 = std::chrono::steady_clock::now();
        int nprompt = (int) ids.size();
        std::vector<int> history = ids;
        int next_tok = 0;

        for (int p = reused; p < nprompt; ++p) {
            const bool is_last = (p == nprompt - 1);
            step_token(ids[p], p, is_last, true);
            state_tokens.push_back(ids[p]);
            if (is_last) {
                next_tok = sample_logits(h_logits, NVOCAB, cfg, history, rng);
            }
        }
        const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
        const int pf_read = nprompt - reused;

        if (json_mode) {
            std::printf("PREFILL {\"prompt_tokens\":%d,\"reused\":%d,\"read\":%d,\"prompt_ms\":%.2f,\"prompt_tps\":%.1f}\n",
                        nprompt, reused, pf_read, pf_ms, pf_ms > 0 ? 1000.0 * pf_read / pf_ms : 0.0);
            std::fflush(stdout);
        } else {
            std::printf("output :");
        }

        const auto t_dec0 = std::chrono::steady_clock::now();
        int generated = 0;
        const char* finish = "length";
        int cur_pos = nprompt;

        for (int gstep = 0; gstep < cfg.max_new; ++gstep) {
            bool is_stop = false;
            for (int st_id : cfg.stops) {
                if (next_tok == st_id) { is_stop = true; break; }
            }
            if (is_stop) {
                finish = "stop";
                break;
            }
            ++generated;
            history.push_back(next_tok);
            if (json_mode) {
                std::printf("TOK %d\n", next_tok);
                std::fflush(stdout);
            } else {
                std::printf(" %d", next_tok);
                std::fflush(stdout);
            }

            if (gstep + 1 >= cfg.max_new || cur_pos >= ctx_max - 1) {
                finish = (cur_pos >= ctx_max - 1) ? "length" : "length";
                break;
            }

            // Feed `next_tok` into the model to advance state and sample the following token
            step_token(next_tok, cur_pos, true, true);
            state_tokens.push_back(next_tok);
            ++cur_pos;
            next_tok = sample_logits(h_logits, NVOCAB, cfg, history, rng);
        }

        const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0).count();
        long long lk1 = 0, vh1 = 0, pr1 = 0;
        for (const auto& s : stages) { lk1 += s.total_lookups; vh1 += s.total_vram_hits; pr1 += s.total_promotions; }
        const long long dlk = lk1 - lk0, dvh = vh1 - vh0, dpr = pr1 - pr0;
        const double hit_rate = dlk > 0 ? (double) dvh / (double) dlk : 1.0;

        if (json_mode) {
            std::printf("DONE {\"generated\":%d,\"prompt_tokens\":%d,\"reused\":%d,\"prompt_ms\":%.2f,\"decode_ms\":%.2f,\"decode_tps\":%.2f,\"vram_hit\":%.4f,\"promoted\":%lld,\"finish\":\"%s\"}\n",
                        generated, nprompt, reused, pf_ms, dec_ms,
                        dec_ms > 0 ? 1000.0 * generated / dec_ms : 0.0,
                        hit_rate, dpr, finish);
            std::fflush(stdout);
        } else {
            std::printf("\nEND\n");
            std::fflush(stdout);
        }
    }
    return 0;
}
