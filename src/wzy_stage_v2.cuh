// wzy_stage_v2.cuh -- Stage Execution for T=1..MAX_W Window Verify + 262,144 (256K) Sparse QSA + INT8 Tiered KV
#pragma once
#include "wzy_qsa_256k.cuh"
#include "wzy_qsa_wmma.cuh"
#include "wzy_cpu_moe.cuh"
#include <cublas_v2.h>
static constexpr int MAX_BATCH = 256; // chunked Tensor-Core prefill batch size!
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s at %s:%d\n", cudaGetErrorString(e_), #x, __FILE__, __LINE__); std::exit(1); } } while (0)

static const char* KD = std::getenv("WZY_MODEL_DIR")
    ? std::getenv("WZY_MODEL_DIR")
    : "/home/liaoyang/models/Maximilian228--Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata";

struct TDesc { std::string n; std::vector<uint64_t> d; uint32_t t; uint64_t o; };
struct Gguf {
    uint8_t* base = nullptr; size_t sz = 0; uint64_t data = 0;
    std::vector<TDesc> ts; std::map<std::string, size_t> idx;
    bool open(const char* p, std::string& err);
    const TDesc* find(const char* n) const { auto i = idx.find(n); return i == idx.end() ? nullptr : &ts[i->second]; }
    const uint8_t* at(const TDesc* t) const { return base + data + t->o; }
};

static inline uint32_t r32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static inline uint64_t r64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
static inline float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float) m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return (s ? -1.f : 1.f) * std::ldexp(1.f + m / 1024.f, (int) e - 15);
}
static inline float bf2f(uint16_t v) { const uint32_t u = (uint32_t) v << 16; float f; std::memcpy(&f, &u, 4); return f; }

inline bool Gguf::open(const char* p, std::string& err) {
    const int fd = ::open(p, O_RDONLY);
    if (fd < 0) { err = std::string("open ") + p; return false; }
    struct stat stt;
    if (fstat(fd, &stt) != 0) { ::close(fd); err = "stat"; return false; }
    sz = (size_t) stt.st_size;
    base = (uint8_t*) mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) { base = nullptr; err = "mmap"; return false; }
    if (std::memcmp(base, "GGUF", 4)) { err = "not gguf"; return false; }
    size_t p2 = 8;
    auto u32 = [&] { const uint32_t v = r32(base + p2); p2 += 4; return v; };
    auto u64 = [&] { const uint64_t v = r64(base + p2); p2 += 8; return v; };
    auto str = [&] { const uint64_t n = u64(); std::string s((char*) base + p2, (size_t) n); p2 += (size_t) n; return s; };
    std::function<void(uint32_t)> skip = [&](uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: p2 += 1; break;
            case 2: case 3: p2 += 2; break;
            case 4: case 5: case 6: p2 += 4; break;
            case 10: case 11: case 12: p2 += 8; break;
            case 8: { const uint64_t n = u64(); p2 += (size_t) n; break; }
            case 9: { const uint32_t et = u32(); const uint64_t n = u64(); for (uint64_t i = 0; i < n; ++i) skip(et); break; }
            default: err = "kv type"; break;
        }
    };
    const uint64_t nt = u64(), nkv = u64();
    for (uint64_t i = 0; i < nkv; ++i) { str(); skip(u32()); }
    ts.resize(nt);
    for (auto& t : ts) {
        t.n = str();
        const uint32_t nd = u32();
        t.d.resize(nd);
        for (auto& x : t.d) x = u64();
        t.t = u32(); t.o = u64();
    }
    data = (p2 + 31) / 32 * 32;
    for (size_t i = 0; i < ts.size(); ++i) idx[ts[i].n] = i;
    return true;
}

struct Dev { float* p = nullptr; size_t n = 0; };
static inline Dev alloc(size_t n) { Dev d; d.n = n; CHECK(cudaMalloc(&d.p, n * 4)); return d; }
static inline Dev alloc0(size_t n) { Dev d = alloc(n); CHECK(cudaMemset(d.p, 0, n * 4)); return d; }

struct Q8 { uint8_t* p = nullptr; unsigned short* s = nullptr; int rows = 0, cols = 0; };
static inline Q8 up_q8_raw(const uint8_t* src, int rows, int cols) {
    Q8 q; q.rows = rows; q.cols = cols;
    const int nb = cols / 32;
    std::vector<uint8_t> codes((size_t) rows * cols);
    std::vector<uint16_t> scales((size_t) rows * nb);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* w = src + (size_t) r * nb * 34;
        for (int b = 0; b < nb; ++b) {
            std::memcpy(&scales[(size_t) r * nb + b], w + b * 34, 2);
            std::memcpy(&codes[(size_t) r * cols + (size_t) b * 32], w + b * 34 + 2, 32);
        }
    }
    CHECK(cudaMalloc(&q.p, (size_t) rows * cols));
    CHECK(cudaMalloc(&q.s, (size_t) rows * nb * 2));
    CHECK(cudaMemcpy(q.p, codes.data(), codes.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(q.s, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    return q;
}
static inline Q8 up_q8(const Gguf& g, const char* name) {
    const TDesc* t = g.find(name);
    if (!t) { std::fprintf(stderr, "missing %s\n", name); std::exit(3); }
    const int cols = (int) t->d[0];
    const int rows = (int) (t->d.size() > 1 ? t->d[1] : 1);
    return up_q8_raw(g.at(t), rows, cols);
}

struct Bf { uint16_t* p = nullptr; int rows = 0, cols = 0; };
static inline Bf up_bf(const Gguf& g, const char* name) {
    const TDesc* t = g.find(name);
    if (!t) { std::fprintf(stderr, "missing %s\n", name); std::exit(3); }
    Bf b;
    b.cols = (int) t->d[0];
    b.rows = (int) (t->d.size() > 1 ? t->d[1] : 1);
    const size_t bytes = (size_t) b.rows * b.cols * 2;
    CHECK(cudaMalloc(&b.p, bytes));
    if (t->t == 30) CHECK(cudaMemcpy(b.p, g.at(t), bytes, cudaMemcpyHostToDevice));
    else {
        std::vector<float> h((size_t) b.rows * b.cols);
        if (t->t == 0) std::memcpy(h.data(), g.at(t), h.size() * 4);
        else { std::fprintf(stderr, "%s type %u\n", name, t->t); std::exit(4); }
        std::vector<uint16_t> o(h.size());
        for (size_t i = 0; i < h.size(); ++i) {
            uint32_t u; std::memcpy(&u, &h[i], 4);
            o[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
        CHECK(cudaMemcpy(b.p, o.data(), bytes, cudaMemcpyHostToDevice));
    }
    return b;
}

static inline Q8 up_bf_as_q8(const Gguf& g, const char* name) {
    Bf b = up_bf(g, name);
    Q8 q; q.rows = b.rows; q.cols = b.cols;
    const size_t n_blocks32 = (size_t) q.rows * (q.cols / 32);
    CHECK(cudaMalloc(&q.p, (size_t) q.rows * q.cols));
    CHECK(cudaMalloc(&q.s, n_blocks32 * sizeof(unsigned short)));
    quant_bf16_to_q8_kernel<<<(n_blocks32 + 7) / 8, 256>>>(b.p, q.p, q.s, n_blocks32);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(b.p));
    return q;
}

static inline Dev up_f32(const Gguf& g, const char* name) {
    const TDesc* t = g.find(name);
    if (!t) { std::fprintf(stderr, "missing %s\n", name); std::exit(3); }
    size_t n = 1;
    for (auto d : t->d) n *= (size_t) d;
    std::vector<float> h(n);
    if (t->t == 30) { const uint16_t* s = (const uint16_t*) g.at(t); for (size_t i = 0; i < n; ++i) h[i] = bf2f(s[i]); }
    else if (t->t == 0) std::memcpy(h.data(), g.at(t), n * 4);
    else if (t->t == 1) { const uint16_t* s = (const uint16_t*) g.at(t); for (size_t i = 0; i < n; ++i) h[i] = h2f(s[i]); }
    else if (t->t == 8) {
        const uint8_t* s = g.at(t);
        const size_t nb = n / 32;
        for (size_t b = 0; b < nb; ++b) {
            const float d = h2f(*(const uint16_t*) (s + b * 34));
            const int8_t* q = (const int8_t*) (s + b * 34 + 2);
            for (int i = 0; i < 32; ++i) h[b * 32 + i] = d * q[i];
        }
    } else { std::fprintf(stderr, "type %u for %s\n", t->t, name); std::exit(4); }
    Dev d = alloc(n);
    CHECK(cudaMemcpy(d.p, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

static inline void fill_tables_current_device() {
    float ue[128];
    for (int x = 0; x < 128; ++x) {
        if (x == 0) { ue[x] = 0.f; continue; }
        const int e = (x >> 3) & 0xF, m = x & 7;
        const float raw = (e == 0) ? std::ldexp((float) m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
        ue[x] = raw * 0.5f;
    }
    unsigned short pr[256];
    const signed char tab[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int b = 0; b < 256; ++b)
        pr[b] = (unsigned short) ((unsigned char) tab[b & 0xF] | ((unsigned char) tab[b >> 4] << 8));
    CHECK(cudaMemcpyToSymbol(kUe4m3, ue, sizeof ue));
    CHECK(cudaMemcpyToSymbol(kPair, pr, sizeof pr));
}

struct LayerW {
    bool qsa = false;
    int qsa_local = -1;
    int gdn_local = -1;
    Q8 qkv, wgate, wout;
    Q8 wq, wk, wv, wo;
    Bf alpha, beta;                 // kept as BF16 (half memory traffic!)
    Dev ssm_a, dtb, conv1d, ssm_norm;
    Dev q_norm, k_norm;
    Bf idx_q, idx_k;                // QSA 4-head Sparse Indexer projections (BF16)
    Dev idx_qn, idx_kn;             // QSA Sparse Indexer norms (F32)
    Dev hc_norm[2];
    Q8 hc_down[2], hc_up[2];
    Bf hc_inj[2];
    Bf g_inp, s_gate_inp;           // kept as BF16 (half memory traffic!)
    Q8 sh_gate, sh_up, sh_down;
};

struct PleSrc {
    uint8_t* base = nullptr;
    size_t sz = 0;
    uint64_t data_off = 0, row_off = 0, rows = 0;
    int row_bytes = 0;
    float scale = 1.f;
    float fp8_lut[256] = {0.f};
    static constexpr uint32_t CACHE_MASK = 524287u; // 512K rows = 84 MB Direct-Mapped PLE Cache!
    struct CacheSlot {
        uint32_t tag = 0xffffffffu;
        uint8_t bytes[160];
    };
    mutable std::vector<CacheSlot> row_cache;
    bool open(const char* path) {
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat stt;
        if (fstat(fd, &stt) != 0) { ::close(fd); return false; }
        sz = (size_t) stt.st_size;
        base = (uint8_t*) mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) { base = nullptr; return false; }
        madvise((void*) base, sz, MADV_RANDOM);
        if (std::memcmp(base, "GGUF", 4) != 0) return false;
        size_t p2 = 8;
        auto u32 = [&] { const uint32_t v = r32(base + p2); p2 += 4; return v; };
        auto u64 = [&] { const uint64_t v = r64(base + p2); p2 += 8; return v; };
        auto str = [&] { const uint64_t n = u64(); std::string s((char*) base + p2, (size_t) n); p2 += (size_t) n; return s; };
        std::function<void(uint32_t)> skip = [&](uint32_t t) {
            switch (t) {
                case 0: case 1: case 7: p2 += 1; break;
                case 2: case 3: p2 += 2; break;
                case 4: case 5: case 6: p2 += 4; break;
                case 10: case 11: case 12: p2 += 8; break;
                case 8: { const uint64_t n = u64(); p2 += (size_t) n; break; }
                case 9: { const uint32_t et = u32(); const uint64_t n = u64(); for (uint64_t i = 0; i < n; ++i) skip(et); break; }
                default: break;
            }
        };
        const uint64_t nt = u64(), nkv = u64();
        for (uint64_t i = 0; i < nkv; ++i) {
            const std::string key = str();
            const uint32_t t = u32();
            if (key == "strata.ple.scale") {
                if (t == 6) { float v; std::memcpy(&v, base + p2, 4); scale = v; }
                else if (t == 12) { double v; std::memcpy(&v, base + p2, 8); scale = (float) v; }
            }
            skip(t);
        }
        for (uint64_t i = 0; i < nt; ++i) {
            const std::string n = str();
            const uint32_t nd = u32();
            std::vector<uint64_t> dd(nd);
            for (auto& x : dd) x = u64();
            u32();
            const uint64_t o = u64();
            if (n == "per_layer_token_embd.weight" && dd.size() >= 2) {
                row_bytes = (int) dd[0];
                rows = dd[1];
                row_off = o;
            }
        }
        data_off = (p2 + 31) / 32 * 32;
        for (int v = 0; v < 256; ++v) {
            const int sg = (v >> 7) & 1, e = (v >> 3) & 0xF, mm = v & 7;
            float f;
            if (e == 0) f = std::ldexp((float) mm, -9);
            else if (e == 15 && mm == 7) f = 0.f;
            else f = std::ldexp(1.f + (float) mm / 8.f, e - 7);
            fp8_lut[v] = (sg ? -f : f) * scale;
        }
        row_cache.resize(CACHE_MASK + 1);
        return rows > 0 && row_bytes == PLE_HEAD;
    }
    inline void prefetch(uint64_t idx) const {
        const uint32_t slot = (uint32_t) ((idx ^ (idx >> 19)) & CACHE_MASK);
        if (row_cache[slot].tag != (uint32_t) idx) {
            const uint8_t* p = base + data_off + row_off + idx * (uint64_t) row_bytes;
            __builtin_prefetch(p, 0, 3);
            __builtin_prefetch(p + 64, 0, 3);
            __builtin_prefetch(p + 128, 0, 3);
        }
    }
    inline void row_direct(uint64_t idx, float* out) const {
        const uint8_t* p = base + data_off + row_off + idx * (uint64_t) row_bytes;
        #pragma unroll
        for (int i = 0; i < PLE_HEAD; ++i) out[i] = fp8_lut[p[i]];
    }
    inline void row(uint64_t idx, float* out) const {
        const uint32_t slot = (uint32_t) ((idx ^ (idx >> 19)) & CACHE_MASK);
        CacheSlot& cs = row_cache[slot];
        if (cs.tag != (uint32_t) idx) {
            const uint8_t* src = base + data_off + row_off + idx * (uint64_t) row_bytes;
            std::memcpy(cs.bytes, src, PLE_HEAD);
            cs.tag = (uint32_t) idx;
        }
        const uint8_t* p = cs.bytes;
        #pragma unroll
        for (int i = 0; i < PLE_HEAD; ++i) out[i] = fp8_lut[p[i]];
    }
    inline void warm_row(uint64_t idx) const {
        const uint32_t slot = (uint32_t) ((idx ^ (idx >> 19)) & CACHE_MASK);
        CacheSlot& cs = row_cache[slot];
        if (cs.tag != (uint32_t) idx) {
            const uint8_t* src = base + data_off + row_off + idx * (uint64_t) row_bytes;
            std::memcpy(cs.bytes, src, PLE_HEAD);
            cs.tag = (uint32_t) idx;
        }
    }
};

__global__ void bcast_emb_to_r_kernel(const float* __restrict__ emb, float* __restrict__ R) {
    const int c = blockIdx.x;
    const int t = blockIdx.y;
    const float* src = emb + (size_t) t * NE;
    float* dst = R + ((size_t) t * HC + c) * NE;
    for (int d = threadIdx.x; d < NE; d += blockDim.x) dst[d] = src[d];
}

struct StageGpu {
    int dev = 0;
    int l_begin = 0, l_end = 25;
    int ctx_max = MAX_CTX_256K;
    int n_qsa = 0, n_gdn = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t abort_stream = nullptr;   // out-of-band stream that flips g_wzy_abort
    // Optional UVA (mapped host) BF16 token embedding: builds the first stage's activation rows on device
    // instead of on the host, which removes the per-window host gather + H2D copy from `enq`.
    const uint16_t* d_emb_table = nullptr;
    int* d_toks = nullptr;
    void set_emb_dev(const uint16_t* t) { d_emb_table = t; }
    cublasHandle_t cublas = nullptr;
    __half *w_fp16 = nullptr, *x_fp16 = nullptr;
    int *grp_slot = nullptr, *grp_cnt = nullptr, *grp_e = nullptr, *d_ngroups = nullptr;

    // Captured graphs for T = 1 (decode), T = 1 body (no head), and speculative verify windows W = 2..MAX_W!
    cudaGraphExec_t exec_body1 = nullptr;
    cudaGraphExec_t exec_full[MAX_W + 1] = {nullptr};

    std::vector<LayerW> L;
    std::vector<Dev> conv, ssm;
    // Per-step snapshots for W > 1 MTP speculative verify so partial acceptance commits in 0.1 ms with ZERO re-run!
    Dev conv_snaps, ssm_snaps, ple_snaps;
    Dev idx_tail_snaps, idx_dead_snaps, idx_pool_snaps;
    Dev R_handoff;
    // 256K QSA Sparse Indexer + INT8 Tiered KV Cache
    Dev cos_tab, sin_tab;
    Dev idx_tail, idx_dead, idx_pooled, idx_scores;
    // Per-layer stride of `idx_pooled` in blocks. Sized for MAX_CTX_256K by default (single-stream
    // engine); the multi-session engine shrinks it to ctx_max/IDX_BLK + 2 so each resident session
    // only pays for the context it can actually reach.
    int idx_pool_stride = MAX_IDX_BLOCKS;
    int *sel_ids = nullptr, *sel_cnt = nullptr;
    signed char *vk_q = nullptr, *vv_q = nullptr;
    unsigned short *vk_s = nullptr, *vv_s = nullptr;
    uint8_t* host_kv_arena = nullptr;
    size_t host_kv_bytes = 0;
    signed char *hk_q = nullptr, *hv_q = nullptr;
    unsigned short *hk_s = nullptr, *hv_s = nullptr;

    // Multi-row [MAX_W, ...] activations & scratch
    Dev R, R_pre_head; // R_pre_head saves final hidden state [MAX_W, HCD] before Output Head for MTP Drafter!
    Dev xn, lo, mixed, mixed2, inject, block;
    Dev qkv, convout, z, decay, bt, on;
    Dev qg, kk, vv, attn, idx_raw, q_idx;
    Dev slog, sids, swts, res_swts, sff, sup, sshared, sgate, tmp_ff;
    Dev parts, comb, ffact, exsc;
    signed char *xq8 = nullptr, *xq4 = nullptr, *eq16 = nullptr, *eq32 = nullptr, *dq16 = nullptr, *dq32 = nullptr;
    float *xs8 = nullptr, *xs4 = nullptr, *es16 = nullptr, *es32 = nullptr, *ds16 = nullptr, *ds32 = nullptr;

    // PLE (Stage 0)
    bool has_ple = false;
    Bf ple_wk, ple_wv;
    Dev ple_nk, ple_nq, ple_nc, ple_cv, ple_hist, d_ple_emb;
    Dev ple_q, ple_k, ple_v, ple_gat, ple_n;

    // Output Head (Stage 1)
    bool has_head = false;
    Dev headnorm, hlogits;
    Q8 head_down, head_up;
    Q8 headq;
    int *h_greedy_ids = nullptr, *d_greedy_ids = nullptr;
    float *h_top16_val = nullptr, *d_top16_val = nullptr;
    int *h_top16_id = nullptr, *d_top16_id = nullptr;
    // WZY_DBG_TOPS layer probe: 3 unsigned ints per layer (routing hash, folded sum, full sum bits).
    unsigned int *h_dbg_sel = nullptr, *d_dbg_sel = nullptr;
    static constexpr int DBG_SEL_SLOTS = 96;
    static constexpr int DBG_SEL_STRIDE = 3;

    // Expert Cache
    int cap = 0, adapt_cap = 2500, static_cap = 0, ring_head = 0;
    uint8_t *xblobs = nullptr, *d_pack = nullptr;
    int* d_lut = nullptr;
    int* h_lut_pinned = nullptr;
    uint8_t *h_adapt_ref = nullptr, *d_adapt_ref = nullptr;
    uint32_t *h_route_freq = nullptr, *d_route_freq = nullptr;
    volatile int *h_skip_flag = nullptr;
    int *d_skip_flag = nullptr;
    std::vector<int> h_lut, ring_owner;
    std::vector<uint32_t> cold_freq;
    std::vector<int> pending_promo;
    int *dpos = nullptr, *dmisses = nullptr, *d_hits = nullptr, *dslots = nullptr, *mkey = nullptr;
    std::vector<int> h_mkey;
    ColdLayerReq *h_cold_req = nullptr, *d_cold_req = nullptr;
    float *h_cold_out[48] = {nullptr}, *d_cold_out[48] = {nullptr};
    uint32_t cold_seq = 0;
    // Stage A (README 7.12): cold-handshake epoch = this window's id base. `d_epoch` is device memory,
    // bumped in-stream at each window end by bump_epoch_kernel, so a kernel can never observe the next
    // window's base. `seq_base` is the host mirror (no GPU read needed) and tracks the old cumulative
    // `cold_seq` value exactly. No `seq` scalar is baked into the kernel arguments any more, which is
    // what lets the window be captured as a CUDA graph (the bump kernel is captured with it).
    unsigned int* d_epoch = nullptr;
    uint32_t seq_base = 0;
    // CUDA graph replay (WZY_GRAPH_STAGE0, README 7.15): one exec per window shape
    // (T, run_head, want_top16, save_snaps); decode only ever produces a handful of shapes.
    std::vector<std::pair<unsigned long long, cudaGraphExec_t>> graph_execs;
    uint32_t* h_toks_g = nullptr;   // pinned, stage-owned: the captured d_toks copy reads this
    long long total_lookups = 0, total_vram_hits = 0, total_promotions = 0;

    void cublas_fp16_gemm(int rows, int cols, int T, const float* x, float* y) {
        f32_to_fp16_vec<<<(((size_t) T * cols + 1) / 2 + 255) / 256, 256, 0, stream>>>(x, x_fp16, (size_t) T * cols);
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &alpha,
                     w_fp16, CUDA_R_16F, cols, x_fp16, CUDA_R_16F, cols,
                     &beta, y, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    void dense_q8(const Q8& W, const float* x, float* y, int rows, int cols, int T) {
        if (T > MAX_W) {
            const size_t total = (size_t) rows * cols;
            dequant_q8p_to_fp16<<<((total + 3) / 4 + 255) / 256, 256, 0, stream>>>(W.p, W.s, w_fp16, rows, cols);
            cublas_fp16_gemm(rows, cols, T, x, y);
            return;
        }
        quant_act_multi<<<((cols * T / 16) + 127) / 128, 128, 0, stream>>>(x, xq8, xs8, nullptr, nullptr, cols, T);
        launch_gemm_q8p(W.p, W.s, xq8, xs8, y, rows, cols, T, stream);
    }

    void dense_bf16(const Bf& W, const float* x, float* y, int rows, int cols, int T) {
        if (T > MAX_W && rows >= 64) {
            const size_t total = (size_t) rows * cols;
            bf16_to_fp16_vec<<<((total + 1) / 2 + 255) / 256, 256, 0, stream>>>(W.p, w_fp16, total);
            cublas_fp16_gemm(rows, cols, T, x, y);
            return;
        }
        launch_gemm_bf16(W.p, x, y, rows, cols, T, stream);
    }

    void gr_read_fused(const Dev& gn, const Q8& gd, const Q8& gu, const Bf& gi,
                       const float* bo_prev, const float* inj_prev,
                       Dev& mx, Dev& inj_out, int T) {
        fused_gr_norm_multi<<<dim3(HC, T), 256, 0, stream>>>(R.p, gn.p, bo_prev, inj_prev, xn.p, NE, HC, EPS, xq8, xs8);
        launch_gr_down_inj_q8(xn.p, xq8, xs8, gd.p, gd.s, gi.p, lo.p, inj_out.p, HCD, HCLR, HC, T, stream);
        quant_act_multi<<<((HCLR * T / 16) + 127) / 128, 128, 0, stream>>>(lo.p, xq8, xs8, nullptr, nullptr, HCLR, T);
        launch_gr_up_mean_q8(xn.p, xq8, xs8, gu.p, gu.s, mx.p, NE, HCLR, HC, T, stream);
    }

    // Record forward pass for `T` tokens (`1 <= T <= MAX_W`).
    // When `save_snaps == true` (`T > 1` speculative verify), GDN & PLE states in global memory are NOT overwritten;
    // instead per-step snapshots [0..T-1] are written to `conv_snaps`, `ssm_snaps`, `ple_snaps` so `commit_window(acc)`
    // commits the exact accepted prefix in 0.1 ms!
    void record_forward(const int* h_pos, const float* h_emb, const float* h_ple,
                        int T, bool run_head, bool save_snaps,
                        const uint8_t* host_pack = nullptr, CpuMoePool* cpu_pool = nullptr, bool want_top16 = false,
                        int snap_offset = 0, bool inline_service = false,
                        StageGpu* peer_s0 = nullptr, uint32_t peer_base = 0, int* peer_l0 = nullptr,
                        int* self_l = nullptr, const int* toks = nullptr) {
        CHECK(cudaSetDevice(dev));
        // WZY_LAYER_PROF=1: time one layer's phases with CUDA events (one-shot, then stays off).
        static const bool lp_on = std::getenv("WZY_LAYER_PROF") != nullptr;
        static cudaEvent_t lp_e[5] = {nullptr};
        static bool lp_ready = false, lp_done = false, lp_pending = false;
        static int lp_T = 0;
        if (lp_on && lp_pending) {
            float ms[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4; ++i) CHECK(cudaEventElapsedTime(&ms[i], lp_e[i], lp_e[i + 1]));
            std::fprintf(stderr,
                         "[layer-prof dev=%d T=%d] gr_read=%.3f attention=%.3f router=%.3f moe=%.3f total=%.3f ms | x48 ~ %.1f ms/window%c",
                         dev, lp_T, ms[0], ms[1], ms[2], ms[3], ms[0] + ms[1] + ms[2] + ms[3],
                         (ms[0] + ms[1] + ms[2] + ms[3]) * 48.0, 10);
            lp_pending = false;
            lp_done = true;
        }
        static long lp_calls = 0;
        static const long lp_at = std::getenv("WZY_LAYER_PROF_AT") ? std::atol(std::getenv("WZY_LAYER_PROF_AT")) : 100;
        const bool lp_here = lp_on && !lp_done && (++lp_calls > lp_at);
        if (lp_on && !lp_ready) {
            for (int i = 0; i < 5; ++i) CHECK(cudaEventCreate(&lp_e[i]));
            lp_ready = true;
        }
        auto LP = [&](int i, int lay) { if (lp_here && lay == l_begin) CHECK(cudaEventRecord(lp_e[i], stream)); };
        const bool use_cpu_pool = (T <= MAX_W && cpu_pool != nullptr && host_pack != nullptr);
        // WZY_COLD_INLINE=0: do not block the host on cold experts between layers. The host enqueues the
        // whole window first, then cold servicing happens in layer order (the GPU spins on
        // wait_and_add_cold_out until each layer's CPU result lands). Measured GPU utilisation during
        // decode is only ~35%, i.e. the launch pipeline is starved by the per-layer host block, so the
        // enqueue-later ordering is the lever. Default 1 keeps the historical behaviour.
        static const bool cold_inline = std::getenv("WZY_COLD_INLINE")
                                      ? std::atoi(std::getenv("WZY_COLD_INLINE")) != 0 : true;
        // ---- CUDA graph replay (WZY_GRAPH_STAGE0, README 7.15) -------------------------------
        const uint32_t seq_base0 = seq_base;   // restored if instantiate fails (see capture close)
        static const bool graph_on = std::getenv("WZY_GRAPH_STAGE0") != nullptr;
        // Each 806-node exec costs tens of MiB of VRAM (see [GRAPH] diagnostics), and the warmup runs
        // wide shapes that decode never produces -- gate capture to the widths decode actually hits.
        static const int graph_maxw = std::getenv("WZY_GRAPH_MAXW") ? std::atoi(std::getenv("WZY_GRAPH_MAXW")) : 4;
        const bool graph_shape = graph_on && use_cpu_pool && toks != nullptr && h_toks_g != nullptr
                                 && T <= graph_maxw;
        unsigned long long gk = 0;
        bool capturing = false, shape_dead = false;
        if (graph_shape) {
            std::memcpy(h_toks_g, toks, (size_t) T * 4);   // captured H2D copy reads this pinned buffer
            gk = ((unsigned long long) T << 8) | (run_head ? 4ull : 0ull) | (want_top16 ? 2ull : 0ull)
               | (save_snaps ? 1ull : 0ull);
            for (const auto& ge : graph_execs) {
                if (ge.first != gk) continue;
                if (ge.second == nullptr) { shape_dead = true; break; }   // no retry after an OOM
                // Replay: host bookkeeping only, then one launch. The epoch bump is INSIDE the capture,
                // so the next window's handshake ids are armed without any host write or stream sync.
                pending_promo.clear();
                seq_base += (uint32_t) (l_end - l_begin);
                cold_seq = seq_base;
                CHECK(cudaGraphLaunch(ge.second, stream));
                return;
            }
            // First window of this shape: capture this call's enqueue sequence, then launch it.
            if (shape_dead) {
                // plain enqueue (and fix up the window-start bookkeeping the replay path skipped)
                if (!use_cpu_pool) CHECK(cudaMemsetAsync(dmisses, 0, 4, stream));
            } else {
                CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                capturing = true;
            }
        }
        if (seq_base > 0xF0000000u) {   // Stage A: recycle handshake ids (never reached in practice)
            CHECK(cudaStreamSynchronize(stream));
            for (int l = l_begin; l < l_end; ++l) {
                h_cold_req[l].ready_seq = 0u;
                h_cold_req[l].cpu_done_seq = 0u;
            }
            _mm_sfence();
            seq_base = 0u;
            cold_seq = 0u;
            if (d_epoch != nullptr) CHECK(cudaMemsetAsync(d_epoch, 0, 4, stream));
        }
        const uint32_t base_seq = seq_base;   // Stage A: the value the kernels read from *d_epoch
        pending_promo.clear();
        if (!use_cpu_pool) CHECK(cudaMemsetAsync(dmisses, 0, 4, stream));
        CHECK(cudaMemcpyAsync(dpos, h_pos, 4, cudaMemcpyHostToDevice, stream));
        if (l_begin == 0) {
            if (d_emb_table != nullptr && d_toks != nullptr && toks != nullptr) {
                CHECK(cudaMemcpyAsync(d_toks, graph_shape ? (const int*) h_toks_g : toks,
                                      (size_t) T * 4, cudaMemcpyHostToDevice, stream));
                dim3 g_emb((NE + 255) / 256, T);
                mtp_gather_bf16_emb<<<g_emb, 256, 0, stream>>>(d_emb_table, d_toks, mixed.p, T);
            } else {
                CHECK(cudaMemcpyAsync(mixed.p, h_emb, (size_t) T * NE * 4, cudaMemcpyHostToDevice, stream));
            }
            bcast_emb_to_r_kernel<<<dim3(HC, T), 256, 0, stream>>>(mixed.p, R.p);
        }
        if (has_ple) {
            CHECK(cudaMemcpyAsync(d_ple_emb.p, h_ple, (size_t) T * NE * 4, cudaMemcpyHostToDevice, stream));
        }

        const float* pending_bo = nullptr;
        const float* pending_inj = nullptr;

        for (int l = l_begin; l < l_end; ++l) {
            LayerW& w = L[l];
            if (l == 1 && has_ple) {
                if (pending_bo) {
                    gr_write_multi<<<dim3(80, T), 256, 0, stream>>>(R.p, pending_bo, pending_inj, NE, HC);
                    pending_bo = nullptr; pending_inj = nullptr;
                }
                ple_group_multi<<<dim3(HC, T), 256, 0, stream>>>(R.p, ple_nq.p, ple_q.p, NE, HC, EPS);
                dense_bf16(ple_wk, d_ple_emb.p, ple_k.p, HCD, NE, T);
                ple_group_multi<<<dim3(HC, T), 256, 0, stream>>>(ple_k.p, ple_nk.p, ple_k.p, NE, HC, EPS);
                dense_bf16(ple_wv, d_ple_emb.p, ple_v.p, NE, NE, T);
                ple_gate_gv_multi<<<dim3(HC, T), 256, 0, stream>>>(ple_k.p, ple_q.p, ple_v.p, ple_gat.p, HC, NE);
                ple_group_multi<<<dim3(HC, T), 256, 0, stream>>>(ple_gat.p, ple_nc.p, ple_n.p, NE, HC, EPS);
                ple_conv_multi<<<(HCD + 255) / 256, 256, 0, stream>>>(
                    ple_n.p, ple_cv.p, ple_hist.p, ple_gat.p, R.p, HCD, PLE_NG, PLE_KERN, T,
                    save_snaps ? (ple_snaps.p + (size_t) snap_offset * HCD * PLE_HIST) : nullptr);
            }

            // Attention half: fold previous layer's FFN gr_write straight into this gr_read!
            LP(0, l);
            gr_read_fused(w.hc_norm[0], w.hc_down[0], w.hc_up[0], w.hc_inj[0], pending_bo, pending_inj, mixed, inject, T);
            LP(1, l);

            if (!w.qsa) {
                if (T > MAX_W) {
                    dense_q8(w.qkv, mixed.p, qkv.p, QKV, NE, T);
                    dense_q8(w.wgate, mixed.p, z.p, GV, NE, T);
                } else {
                    quant_act_multi<<<((NE * T / 16) + 127) / 128, 128, 0, stream>>>(mixed.p, xq8, xs8, nullptr, nullptr, NE, T);
                    launch_gemm_q8p(w.qkv.p, w.qkv.s, xq8, xs8, qkv.p, QKV, NE, T, stream);
                    launch_gemm_q8p(w.wgate.p, w.wgate.s, xq8, xs8, z.p, GV, NE, T, stream);
                }
                float* c_snap = save_snaps ? (conv_snaps.p + ((size_t) w.gdn_local * MAX_W + snap_offset) * QKV * 3) : nullptr;
                float* s_snap = save_snaps ? (ssm_snaps.p + ((size_t) w.gdn_local * MAX_W + snap_offset) * VH * ST * ST) : nullptr;

                fused_gdn_conv_l2_multi<<<QKV / 128, 128, 0, stream>>>(
                    conv[l].p, qkv.p, w.conv1d.p, convout.p, QKV, 2 * GH, EPS, T, c_snap);
                fused_gdn_ab_multi<<<(2 * VH + 7) / 8, 256, 0, stream>>>(
                    mixed.p, w.alpha.p, w.beta.p, w.dtb.p, w.ssm_a.p, decay.p, bt.p, NE, VH, T);
                fused_gdn_step_norm_multi<<<VH, dim3(128, 4), 0, stream>>>(
                    ssm[l].p, convout.p, decay.p, bt.p, z.p, w.ssm_norm.p, EPS, on.p, GH, VH, T, s_snap);
                dense_q8(w.wout, on.p, block.p, NE, GV, T);
            } else {
                const int ql = w.qsa_local;
                if (T > MAX_W) {
                    dense_q8(w.wq, mixed.p, qg.p, 2 * QD, NE, T);
                    dense_q8(w.wk, mixed.p, kk.p, KVD, NE, T);
                    dense_q8(w.wv, mixed.p, vv.p, KVD, NE, T);
                } else {
                    quant_act_multi<<<((NE * T / 16) + 127) / 128, 128, 0, stream>>>(mixed.p, xq8, xs8, nullptr, nullptr, NE, T);
                    launch_gemm_q8p(w.wq.p, w.wq.s, xq8, xs8, qg.p, 2 * QD, NE, T, stream);
                    launch_gemm_q8p(w.wk.p, w.wk.s, xq8, xs8, kk.p, KVD, NE, T, stream);
                    launch_gemm_q8p(w.wv.p, w.wv.s, xq8, xs8, vv.p, KVD, NE, T, stream);
                }

                dense_bf16(w.idx_k, mixed.p, idx_raw.p, IDX_DIM, NE, T);
                dense_bf16(w.idx_q, mixed.p, q_idx.p, IDX_HEADS * IDX_DIM, NE, T);
                fused_head_norm_rope_multi<<<dim3(QH, T), 128, 0, stream>>>(
                    qg.p, w.q_norm.p, cos_tab.p, sin_tab.p, dpos, QH, HD, 2 * HD, 2 * QD, EPS);
                fused_head_norm_rope_multi<<<dim3(KVH, T), 128, 0, stream>>>(
                    kk.p, w.k_norm.p, cos_tab.p, sin_tab.p, dpos, KVH, HD, HD, KVD, EPS);
                fused_head_norm_rope_multi<<<dim3(IDX_HEADS, T), 128, 0, stream>>>(
                    q_idx.p, w.idx_qn.p, cos_tab.p, sin_tab.p, dpos, IDX_HEADS, IDX_DIM, IDX_DIM, IDX_HEADS * IDX_DIM, EPS);

                signed char* l_vkq = vk_q + (size_t) ql * KV_VRAM_RING * KVD;
                unsigned short* l_vks = vk_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL;
                signed char* l_vvq = vv_q + (size_t) ql * KV_VRAM_RING * KVD;
                unsigned short* l_vvs = vv_s + (size_t) ql * KV_VRAM_RING * KV_SC_PER_CELL;

                signed char* l_hkq = hk_q ? (hk_q + (size_t) ql * ctx_max * KVD) : nullptr;
                unsigned short* l_hks = hk_s ? (hk_s + (size_t) ql * ctx_max * KV_SC_PER_CELL) : nullptr;
                signed char* l_hvq = hv_q ? (hv_q + (size_t) ql * ctx_max * KVD) : nullptr;
                unsigned short* l_hvs = hv_s ? (hv_s + (size_t) ql * ctx_max * KV_SC_PER_CELL) : nullptr;

                kv_append_int8_multi<<<dim3(8, T), 64, 0, stream>>>(
                    kk.p, vv.p, l_vkq, l_vks, l_vvq, l_vvs, l_hkq, l_hks, l_hvq, l_hvs, dpos, KV_VRAM_RING, ctx_max);

                float* l_tail = idx_tail.p + (size_t) ql * (IDX_BLK - 1) * IDX_DIM;
                float* l_dead = idx_dead.p + (size_t) ql * IDX_DIM;
                float* l_pool = idx_pooled.p + (size_t) ql * idx_pool_stride * IDX_DIM;

                float* t_snap = (save_snaps && idx_tail_snaps.p) ? (idx_tail_snaps.p + ((size_t) ql * MAX_W + snap_offset) * (IDX_BLK - 1) * IDX_DIM) : nullptr;
                float* d_snap = (save_snaps && idx_dead_snaps.p) ? (idx_dead_snaps.p + ((size_t) ql * MAX_W + snap_offset) * IDX_DIM) : nullptr;
                float* p_snap = (save_snaps && idx_pool_snaps.p) ? (idx_pool_snaps.p + ((size_t) ql * MAX_W + snap_offset) * 2 * IDX_DIM) : nullptr;
                qsa_indexer_append_multi<<<1, 128, 0, stream>>>(
                    idx_raw.p, w.idx_kn.p, cos_tab.p, sin_tab.p, dpos, l_tail, l_dead, l_pool, ctx_max, EPS, T,
                    t_snap, d_snap, p_snap);

                const int max_b = std::min(ctx_max / IDX_BLK + 2, (*h_pos + T) / IDX_BLK + 4);
                if (*h_pos + T > IDX_TOPK) {
                    pf_qsa_block_scores_wmma<<<dim3((max_b + 63) / 64, 1), 256, 0, stream>>>(
                        l_pool, q_idx.p, dpos, max_b, idx_scores.p, 0, T);
                    pf_qsa_tail_score_kernel<<<T, 128, 0, stream>>>(
                        l_dead, q_idx.p, dpos, max_b, idx_scores.p, 0);
                }
                qsa_block_topk_multi<<<T, 256, 0, stream>>>(
                    idx_scores.p, dpos, max_b, sel_ids, sel_cnt);

                if (*h_pos + T > KV_VRAM_RING && l_hkq != nullptr) {
                    // Above 32K context: GQA-12 Tensor-Core kernel reads each PCIe host KV cell ONCE for all 12 query heads (12x less PCIe traffic!)
                    pf_qsa_wmma_sparse_attn<<<dim3(KVH, T), 256, 0, stream>>>(
                        qg.p, l_vkq, l_vks, l_vvq, l_vvs, l_hkq, l_hks, l_hvq, l_hvs, sel_ids, dpos, nullptr, KV_VRAM_RING, attn.p);
                } else {
                    qsa_attend_int8_sparse_multi<<<dim3(QH, T), 256, 0, stream>>>(
                        qg.p, l_vkq, l_vks, l_vvq, l_vvs, l_hkq, l_hks, l_hvq, l_hvs, sel_ids, dpos, attn.p, KV_VRAM_RING);
                }
                dense_q8(w.wo, attn.p, block.p, NE, QD, T);
            }

            // FFN half: fold Attention's gr_write straight into FFN's gr_read!
            // Note: `inject` holds Attention's inject; save to `ple_q` (unused scratch) or write before overwriting `inject`
            LP(2, l);
            gr_write_multi<<<dim3(80, T), 256, 0, stream>>>(R.p, block.p, inject.p, NE, HC);
            gr_read_fused(w.hc_norm[1], w.hc_down[1], w.hc_up[1], w.hc_inj[1], nullptr, nullptr, mixed2, inject, T);

            // Quantize `mixed2` ONCE into BOTH 32-wide (for shared expert) AND 16-wide (for NVFP4 routed experts)!
            quant_act_multi<<<((NE * T / 16) + 127) / 128, 128, 0, stream>>>(mixed2.p, eq32, es32, eq16, es16, NE, T);

            dense_bf16(w.g_inp, mixed2.p, slog.p, NEXP, NE, T);
            // F1b: warp 1 of the top-10 kernel also computes the rows=1 sgate gemm (saves a launch)
            router_top10_fast_multi<<<T, 256, 0, stream>>>(slog.p, (int*) sids.p, swts.p,
                                                            w.s_gate_inp.p, mixed2.p, sgate.p, NE);
            LP(3, l);
            const int total_k = T * TOPK;
            if (T <= MAX_W && cpu_pool != nullptr && host_pack != nullptr) {
                split_route_hit_cold_kernel<<<1, 512, 0, stream>>>(
                    (const int*) sids.p, swts.p, T, d_lut, l, dslots, res_swts.p,
                    grp_slot, grp_cnt, grp_e, d_ngroups, eq16, es16, d_cold_req + l,
                    d_epoch, l - l_begin, static_cap, d_adapt_ref, mkey + l);

                launch_gemm_q8p(w.sh_gate.p, w.sh_gate.s, eq32, es32, sff.p, NFF, NE, T, stream);
                launch_gemm_q8p(w.sh_up.p, w.sh_up.s, eq32, es32, sup.p, NFF, NE, T, stream);
                silu_mul_quant32_multi<<<((NFF * T / 32) + 127) / 128, 128, 0, stream>>>(sff.p, sup.p, xq8, xs8, (NFF / 32) * T);
                launch_gemm_q8p(w.sh_down.p, w.sh_down.s, xq8, xs8, sshared.p, NE, NFF, T, stream);
                launch_gemm_bf16(w.s_gate_inp.p, mixed2.p, sgate.p, 1, NE, T, stream);

                exp_gu_quant_grouped<<<dim3(NFF / 16, total_k), 256, 0, stream>>>(
                    xblobs, grp_slot, grp_cnt, grp_e, d_ngroups, eq16, es16, dq16, ds16);
                exp_down_2r_grouped<<<dim3(NE / 64, total_k), 256, 0, stream>>>(
                    xblobs, grp_slot, grp_cnt, grp_e, d_ngroups, dq16, ds16, parts.p);
                moe_combine_shared_multi<<<dim3((NE + 255) / 256, T), 256, 0, stream>>>(
                    parts.p, res_swts.p, sshared.p, sgate.p, comb.p, NE, TOPK);
                wait_and_add_cold_out_kernel<<<(T * NE + 255) / 256, 256, 0, stream>>>(
                    comb.p, d_cold_out[l], d_cold_req + l, d_epoch, l - l_begin, T * NE, mkey + l);
            } else {
                if (T > MAX_W) {
                    dense_q8(w.sh_gate, mixed2.p, sff.p, NFF, NE, T);
                    dense_q8(w.sh_up, mixed2.p, sup.p, NFF, NE, T);
                } else {
                    launch_gemm_q8p(w.sh_gate.p, w.sh_gate.s, eq32, es32, sff.p, NFF, NE, T, stream);
                    launch_gemm_q8p(w.sh_up.p, w.sh_up.s, eq32, es32, sup.p, NFF, NE, T, stream);
                }
                silu_mul_multi<<<(NFF * T + 255) / 256, 256, 0, stream>>>(sff.p, sup.p, tmp_ff.p, NFF * T);
                dense_q8(w.sh_down, tmp_ff.p, sshared.p, NE, NFF, T);
                // F1b: sgate computed inside router_top10_fast_multi (warp 1)

                map_slots_lut_multi<<<(total_k + 31) / 32, 32, 0, stream>>>((const int*) sids.p, total_k, d_lut, l, dslots, dmisses, mkey);
                if (T == 1) {
                    exp_scales_multi<<<(total_k + 31) / 32, 32, 0, stream>>>(xblobs, d_pack, dslots, total_k, exsc.p);
                    exp_gu_vec<<<(total_k * NFF + 7) / 8, 256, 0, stream>>>(xblobs, d_pack, dslots, total_k, eq16, es16, ffact.p, exsc.p);
                    quant_act_multi<<<((total_k * NFF / 16) + 127) / 128, 128, 0, stream>>>(ffact.p, nullptr, nullptr, dq16, ds16, NFF, total_k);
                    exp_down_vec<<<(total_k * NE + 7) / 8, 256, 0, stream>>>(xblobs, d_pack, dslots, total_k, dq16, ds16, parts.p, exsc.p);
                } else {
                    group_experts_kernel<<<1, 512, 0, stream>>>((const int*) sids.p, dslots, total_k, grp_slot, grp_cnt, grp_e, d_ngroups);
                    exp_gu_grouped<<<(total_k * NFF + 7) / 8, 256, 0, stream>>>(
                        xblobs, d_pack, grp_slot, grp_cnt, grp_e, d_ngroups, total_k, eq16, es16, ffact.p, nullptr);
                    quant_act_multi<<<((total_k * NFF / 16) + 127) / 128, 128, 0, stream>>>(ffact.p, nullptr, nullptr, dq16, ds16, NFF, total_k);
                    exp_down_grouped<<<(total_k * NE + 31) / 32, 256, 0, stream>>>(
                        xblobs, d_pack, grp_slot, grp_cnt, grp_e, d_ngroups, total_k, dq16, ds16, parts.p, nullptr);
                }
                moe_combine_shared_multi<<<dim3((NE + 255) / 256, T), 256, 0, stream>>>(
                    parts.p, swts.p, sshared.p, sgate.p, comb.p, NE, TOPK);
            }
            // Width-invariance probe (WZY_DBG_TOPS): signature of row 0 only, so the comparison
            // between W=1 and W>1 is about row 0's arithmetic, never about the extra rows.
            static const bool dbg_sel = std::getenv("WZY_DBG_TOPS") != nullptr;
            if (dbg_sel && d_dbg_sel != nullptr && T >= 1 && (l - l_begin) < DBG_SEL_SLOTS) {
                dbg_probe_layer<<<1, 1, 0, stream>>>((const int*) sids.p, comb.p, TOPK, NE,
                                                    d_dbg_sel + (size_t) (l - l_begin) * DBG_SEL_STRIDE);
            }
            if (lp_here && l == l_begin) {
                LP(4, l);
                lp_T = T;
                lp_pending = true;      // computed on the next call: syncing here would deadlock (GPU waits for CPU cold experts)
            }
            pending_bo = comb.p;
            pending_inj = inject.p;
            if (inline_service && use_cpu_pool && cold_inline && l > l_begin) {
                service_one_cold_layer(l - 1, host_pack, cpu_pool, base_seq, T, false);
            }
            if (peer_s0 != nullptr && peer_l0 != nullptr && use_cpu_pool) {
                while (*peer_l0 < peer_s0->l_end) {
                    const uint32_t want0 = peer_base + (uint32_t) (*peer_l0 - peer_s0->l_begin) + 1u;
                    if (peer_s0->h_cold_req[*peer_l0].ready_seq != want0) break;
                    peer_s0->service_one_cold_layer(*peer_l0, host_pack, cpu_pool, peer_base, T, false);
                    ++(*peer_l0);
                }
            }
            if (self_l != nullptr && use_cpu_pool && (peer_s0 == nullptr || peer_l0 == nullptr || *peer_l0 >= peer_s0->l_end)) {
                while (*self_l <= l) {
                    const uint32_t want_s = base_seq + (uint32_t) (*self_l - l_begin) + 1u;
                    if (h_cold_req[*self_l].ready_seq != want_s) break;
                    service_one_cold_layer(*self_l, host_pack, cpu_pool, base_seq, T, false);
                    ++(*self_l);
                }
            }
        }

        // Stage A: arm the NEXT window's handshake epoch. Stream-ordered after every handshake kernel of
        // this window, so their ids are already fixed; the value advances exactly like the old
        // per-layer `++cold_seq` (one window's worth of layer ids, cold-branch windows only).
        if (use_cpu_pool) {
            const unsigned int step = (unsigned int) (l_end - l_begin);
            if (d_epoch != nullptr) bump_epoch_kernel<<<1, 1, 0, stream>>>(d_epoch, step);
            seq_base += step;
            cold_seq = seq_base;   // the base the engine reads for the NEXT window
        }

        if (pending_bo) {
            gr_write_multi<<<dim3(80, T), 256, 0, stream>>>(R.p, pending_bo, pending_inj, NE, HC);
            pending_bo = nullptr; pending_inj = nullptr;
        }

        if (has_head && run_head) {
            // Save pre-head residual R [T, HCD] for MTP Drafter!
            CHECK(cudaMemcpyAsync(R_pre_head.p, R.p, (size_t) T * HCD * 4, cudaMemcpyDeviceToDevice, stream));
            Dev dummy_inj; dummy_inj.p = nullptr;
            Bf no_inj; no_inj.p = nullptr;
            gr_read_fused(headnorm, head_down, head_up, no_inj, nullptr, nullptr, mixed, dummy_inj, T);
            dense_q8(headq, mixed.p, hlogits.p, NVOCAB, NE, T);
            if (d_greedy_ids) {
                if (want_top16) row_top16_multi_kernel<<<T, 512, 0, stream>>>(hlogits.p, NVOCAB, d_greedy_ids, d_top16_val, d_top16_id);
                else row_argmax_multi_kernel<<<T, 1024, 0, stream>>>(hlogits.p, NVOCAB, d_greedy_ids);
            }
        }
        if (inline_service && use_cpu_pool && l_end > l_begin) {
            if (cold_inline) {
                service_one_cold_layer(l_end - 1, host_pack, cpu_pool, base_seq, T, false);
            } else {
                // Enqueue-first mode: everything after the first layer is still pending, so drain the
                // whole stage in layer order (the GPU is spinning on the earliest unserved layer).
                for (int l = l_begin; l < l_end; ++l)
                    service_one_cold_layer(l, host_pack, cpu_pool, base_seq, T, false);
            }
        }

        if (capturing) {
            cudaGraph_t g = nullptr;
            CHECK(cudaStreamEndCapture(stream, &g));
            size_t nn = 0;
            cudaGraphGetNodes(g, nullptr, &nn);
            cudaGraphExec_t ex = nullptr;
            cudaError_t ie = cudaGraphInstantiate(&ex, g, 0);
            if (ie == cudaSuccess) {
                graph_execs.push_back({gk, ex});
                std::fprintf(stderr, "[GRAPH] shape=%llx nodes=%zu instantiated%c",
                             (unsigned long long) gk, nn, (char) 10);
                cudaGraphDestroy(g);
                CHECK(cudaGraphLaunch(ex, stream));   // the capture window must still execute
            } else {
                graph_execs.push_back({gk, nullptr});   // never retry this shape
                cudaGraphDestroy(g);
                size_t fr = 0, tt = 0;
                cudaMemGetInfo(&fr, &tt);
                std::fprintf(stderr, "[GRAPH] shape=%llx nodes=%zu instantiate FAILED (%s) free=%.1f MiB -- enqueue fallback%c",
                             (unsigned long long) gk, nn, cudaGetErrorString(ie), (double) fr / 1048576.0, (char) 10);
                // The captured call never executed (d_epoch NOT bumped) but already advanced the host
                // bookkeeping: undo it and re-run the body once, landing on the plain-enqueue path.
                seq_base = seq_base0;
                cold_seq = seq_base0;
                record_forward(h_pos, h_emb, h_ple, T, run_head, save_snaps, host_pack, cpu_pool,
                               want_top16, snap_offset, inline_service, peer_s0, peer_base, peer_l0,
                               self_l, toks);
                return;
            }
        }
    }


    void service_one_cold_layer(int l, const uint8_t* host_pack, CpuMoePool* cpu_pool, uint32_t base_seq, int T, bool skip_compute = false) {
        const int total_k = T * TOPK;
        const uint32_t seq = base_seq + (uint32_t) (l - l_begin) + 1u;
        ColdLayerReq* req = h_cold_req + l;
        while (req->ready_seq != seq) _mm_pause();
        _mm_lfence();
        std::atomic_thread_fence(std::memory_order_acquire);
        if (skip_compute) {
            if (req->n_cold > 0) {
                _mm_sfence();
                std::atomic_thread_fence(std::memory_order_seq_cst);
                req->cpu_done_seq = seq;
            }
            return;
        }
        total_lookups += total_k;
        int l_miss = 0;
        if (req->n_cold > 0) {
            for (int cg = 0; cg < req->n_cold; ++cg) {
                const int cnt_cg = req->cold_cnt[cg];
                l_miss += cnt_cg;
                if (adapt_cap > 0 && !cold_freq.empty()) {
                    const int gk = req->cold_gkey[cg];
                    if (gk >= l_begin * NEXP && gk < l_end * NEXP) {
                        // Only promote experts routed by confirmed token 0 or >=2 tokens in the window (prevents rejected drafts from polluting VRAM!)
                        const bool confirmed_use = (cnt_cg >= 2 || req->cold_tok[cg][0] == 0);
                        const uint32_t f = confirmed_use ? (cold_freq[gk] += (uint32_t) cnt_cg) : cold_freq[gk];
                        static const uint32_t env_min_f = std::getenv("WZY_PROMO_MIN_F") ? (uint32_t) std::max(1, std::atoi(std::getenv("WZY_PROMO_MIN_F"))) : 4u;
                        const uint32_t min_f = env_min_f;
                        // README 7.13: push EVERY eligible candidate. The old `pending_promo.size() < max_q`
                        // cutoff here made the queued subset depend on the kernel's nondeterministic
                        // atomicAdd service order -- the root cause of irreproducible greedy decode.
                        // The queue cap is applied deterministically in promote_cold after sorting.
                        if (confirmed_use && f >= min_f && h_lut[gk] < 0) {
                            pending_promo.push_back(gk);
                        }
                    }
                }
            }
            // Deterministic expert order. split_route_hit_cold_kernel assigns cg with atomicAdd, so
            // the group order varies run to run; float addition is not associative, which made greedy
            // decode non-reproducible (same prompt, same config, diverging token streams). Sorting by
            // gkey gives a canonical accumulation order with no effect on precision.
            if (req->n_cold > 1) {
                for (int i = 1; i < req->n_cold; ++i) {
                    const int kg = req->cold_gkey[i];
                    const int kc = req->cold_cnt[i];
                    int kt[GRP_MAX]; float kw[GRP_MAX];
                    for (int m = 0; m < GRP_MAX; ++m) { kt[m] = req->cold_tok[i][m]; kw[m] = req->cold_wt[i][m]; }
                    int j = i - 1;
                    while (j >= 0 && req->cold_gkey[j] > kg) {
                        req->cold_gkey[j + 1] = req->cold_gkey[j];
                        req->cold_cnt[j + 1] = req->cold_cnt[j];
                        for (int m = 0; m < GRP_MAX; ++m) {
                            req->cold_tok[j + 1][m] = req->cold_tok[j][m];
                            req->cold_wt[j + 1][m] = req->cold_wt[j][m];
                        }
                        --j;
                    }
                    req->cold_gkey[j + 1] = kg;
                    req->cold_cnt[j + 1] = kc;
                    for (int m = 0; m < GRP_MAX; ++m) { req->cold_tok[j + 1][m] = kt[m]; req->cold_wt[j + 1][m] = kw[m]; }
                }
            }
            cpu_pool->run(req, host_pack, h_cold_out[l]);
            _mm_sfence();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            req->cpu_done_seq = seq;
        }
        total_vram_hits += (total_k - l_miss);
    }

    void service_cold_layers(const uint8_t* host_pack, CpuMoePool* cpu_pool, uint32_t base_seq, int T, bool skip_compute = false, int start_l = -1) {
        if (T > MAX_W || cpu_pool == nullptr || host_pack == nullptr) return;
        for (int l = (start_l >= l_begin ? start_l : l_begin); l < l_end; ++l) {
            service_one_cold_layer(l, host_pack, cpu_pool, base_seq, T, skip_compute);
        }
    }
    // Commit speculative window state at index `acc_idx` (`0 <= acc_idx < W`, where `accepted_count = acc_idx + 1`)
    void commit_window_state(int acc_idx, int acc_pos = -1) {
        CHECK(cudaSetDevice(dev));
        for (int l = l_begin; l < l_end; ++l) {
            LayerW& w = L[l];
            if (w.qsa) continue;
            const float* cs = conv_snaps.p + ((size_t) w.gdn_local * MAX_W + acc_idx) * (QKV * 3);
            const float* ss = ssm_snaps.p + ((size_t) w.gdn_local * MAX_W + acc_idx) * (VH * ST * ST);
            CHECK(cudaMemcpyAsync(conv[l].p, cs, (size_t) QKV * 3 * 4, cudaMemcpyDeviceToDevice, stream));
            CHECK(cudaMemcpyAsync(ssm[l].p, ss, (size_t) VH * ST * ST * 4, cudaMemcpyDeviceToDevice, stream));
        }
        if (has_ple) {
            const float* ps = ple_snaps.p + (size_t) acc_idx * (HCD * PLE_HIST);
            CHECK(cudaMemcpyAsync(ple_hist.p, ps, (size_t) HCD * PLE_HIST * 4, cudaMemcpyDeviceToDevice, stream));
        }
        if (acc_pos >= 0 && n_qsa > 0 && idx_tail_snaps.p) {
            qsa_restore_snap_kernel<<<n_qsa, 128, 0, stream>>>(
                idx_tail.p, idx_dead.p, idx_pooled.p,
                idx_tail_snaps.p, idx_dead_snaps.p, idx_pool_snaps.p, acc_idx, acc_pos);
        }
    }

    void promote_cold(const uint8_t* host_pack_raw, int T, bool do_promote) {
        // WZY_FREEZE_CACHE: no eviction/promotion at all, so the cold/hot expert split stays identical between runs.
        // Cold experts take the CPU AVX2 path and hot ones the GPU NVFP4 path, which round differently; freezing the
        // cache is therefore the only way two runs can be compared token-for-token (A/B of engine changes).
        static const bool cache_frozen = std::getenv("WZY_FREEZE_CACHE") != nullptr;
        if (cache_frozen) {
            pending_promo.clear();
            return;
        }
        if (T <= MAX_W) {
            if (!do_promote || adapt_cap <= 0 || pending_promo.empty()) {
                pending_promo.clear();
                return;
            }
            CHECK(cudaSetDevice(dev));
            int promoted = 0;
            static const int env_max_promo = std::getenv("WZY_PROMO_PER_WIN") ? std::max(1, std::atoi(std::getenv("WZY_PROMO_PER_WIN"))) : 8;
            static const int env_max_scan = std::getenv("WZY_PROMO_SCAN") ? std::max(16, std::atoi(std::getenv("WZY_PROMO_SCAN"))) : 256;
            const int max_promo = env_max_promo;
            const int max_scans = std::min(adapt_cap, env_max_scan);
            // README 7.13: deterministic priority. `pending_promo` is in nondeterministic atomicAdd
            // service order; sorting by (freq desc, gkey asc) makes the capped subset a pure function
            // of this window's routed groups. Same max_q budget as before (WZY_PROMO_QUEUE).
            {
                static const int env_max_q = std::getenv("WZY_PROMO_QUEUE") ? std::max(1, std::atoi(std::getenv("WZY_PROMO_QUEUE"))) : 16;
                const int max_q = env_max_q;
                std::sort(pending_promo.begin(), pending_promo.end(),
                          [&](int a, int b) {
                              const uint32_t fa = cold_freq[a], fb = cold_freq[b];
                              if (fa != fb) return fa > fb;   // hottest first
                              return a < b;                   // stable, value-independent tie-break
                          });
                if ((int) pending_promo.size() > max_q) pending_promo.resize(max_q);
            }
            for (int gkey : pending_promo) {
                if (promoted >= max_promo) break;
                if (gkey < l_begin * NEXP || gkey >= l_end * NEXP || h_lut[gkey] >= 0) continue;
                int scans = 0;
                while (h_adapt_ref != nullptr && scans < max_scans) {
                    const int cand = ring_head % adapt_cap;
                    if (h_adapt_ref[cand] == 0) break;
                    h_adapt_ref[cand] = 0;
                    ++ring_head;
                    ++scans;
                }
                if (scans == max_scans && cold_freq[gkey] < 24u) continue;
                const int r = (ring_head++) % adapt_cap;
                const int vslot = static_cap + r;
                const int old_k = ring_owner[r];
                if (old_k >= 0) {
                    h_lut[old_k] = -1;
                    if (h_lut_pinned) h_lut_pinned[old_k] = -1;
                    CHECK(cudaMemcpyAsync(d_lut + old_k, &h_lut[old_k], 4, cudaMemcpyHostToDevice, stream));
                }
                CHECK(cudaMemcpyAsync(xblobs + (size_t) vslot * PER,
                                      host_pack_raw + (size_t) gkey * PER,
                                      PER, cudaMemcpyHostToDevice, stream));
                ring_owner[r] = gkey;
                h_lut[gkey] = vslot;
                if (h_lut_pinned) h_lut_pinned[gkey] = vslot;
                CHECK(cudaMemcpyAsync(d_lut + gkey, &h_lut[gkey], 4, cudaMemcpyHostToDevice, stream));
                if (h_adapt_ref) h_adapt_ref[r] = 1;
                cold_freq[gkey] = 0;
                ++promoted;
                ++total_promotions;
            }
            pending_promo.clear();
            return;
        }
        if (adapt_cap <= 0) return;
    }
    void reset_state() {
        CHECK(cudaSetDevice(dev));
        for (int l = l_begin; l < l_end; ++l) {
            if (conv[l].p) CHECK(cudaMemsetAsync(conv[l].p, 0, (size_t) QKV * 3 * 4, stream));
            if (ssm[l].p) CHECK(cudaMemsetAsync(ssm[l].p, 0, (size_t) VH * ST * ST * 4, stream));
        }
        if (n_qsa > 0) {
            CHECK(cudaMemsetAsync(idx_tail.p, 0, idx_tail.n * 4, stream));
            CHECK(cudaMemsetAsync(idx_dead.p, 0, idx_dead.n * 4, stream));
        }
        if (has_ple && ple_hist.p)
            CHECK(cudaMemsetAsync(ple_hist.p, 0, (size_t) HCD * PLE_HIST * 4, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
};
