// wzy_pp2_engine.cu -- Dual-GPU (2x RTX 2080 Ti 22GB SM75) PP2 Pipeline Engine for Qwen3.8-Flash-Next-NVFP4
#include "wzy_kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s at line %d\n", cudaGetErrorString(e_), #x, __LINE__); std::exit(1); } } while (0)

static const char* KD = "/home/liaoyang/models/Maximilian228--Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata";

struct T { std::string n; std::vector<uint64_t> d; uint32_t t; uint64_t o; };
struct Gguf {
    uint8_t* base = nullptr; size_t sz = 0; uint64_t data = 0;
    std::vector<T> ts; std::map<std::string, size_t> idx;
    bool open(const char* p, std::string& err);
    const T* find(const char* n) const { auto i = idx.find(n); return i == idx.end() ? nullptr : &ts[i->second]; }
    const uint8_t* at(const T* t) const { return base + data + t->o; }
};
static uint32_t r32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static uint64_t r64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
static float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float) m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return (s ? -1.f : 1.f) * std::ldexp(1.f + m / 1024.f, (int) e - 15);
}
static float bf2f(uint16_t v) { const uint32_t u = (uint32_t) v << 16; float f; std::memcpy(&f, &u, 4); return f; }

bool Gguf::open(const char* p, std::string& err) {
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
static Dev alloc(size_t n) { Dev d; d.n = n; CHECK(cudaMalloc(&d.p, n * 4)); return d; }
static Dev alloc0(size_t n) { Dev d = alloc(n); CHECK(cudaMemset(d.p, 0, n * 4)); return d; }

struct Q8 { uint8_t* p = nullptr; unsigned short* s = nullptr; int rows = 0, cols = 0; };
static Q8 up_q8(const Gguf& g, const char* name) {
    const T* t = g.find(name);
    if (!t) { std::fprintf(stderr, "missing %s\n", name); std::exit(3); }
    Q8 q;
    q.cols = (int) t->d[0];
    q.rows = (int) (t->d.size() > 1 ? t->d[1] : 1);
    const int nb = q.cols / 32;
    std::vector<uint8_t> codes((size_t) q.rows * q.cols);
    std::vector<uint16_t> scales((size_t) q.rows * nb);
    const uint8_t* src = g.at(t);
    for (int r = 0; r < q.rows; ++r) {
        const uint8_t* w = src + (size_t) r * nb * 34;
        for (int b = 0; b < nb; ++b) {
            std::memcpy(&scales[(size_t) r * nb + b], w + b * 34, 2);
            std::memcpy(&codes[(size_t) r * q.cols + (size_t) b * 32], w + b * 34 + 2, 32);
        }
    }
    CHECK(cudaMalloc(&q.p, (size_t) q.rows * q.cols));
    CHECK(cudaMalloc(&q.s, (size_t) q.rows * nb * 2));
    CHECK(cudaMemcpy(q.p, codes.data(), codes.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(q.s, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    return q;
}

struct Bf { uint16_t* p = nullptr; int rows = 0, cols = 0; };
static Bf up_bf(const Gguf& g, const char* name) {
    const T* t = g.find(name);
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

static Dev up_f32(const Gguf& g, const char* name) {
    const T* t = g.find(name);
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

static void fill_tables_current_device() {
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
    Q8 qkv, wgate, wout;
    Q8 wq, wk, wv, wo;
    Dev alpha, beta, ssm_a, dtb, conv1d, ssm_norm;
    Dev q_norm, k_norm;
    Dev hc_norm[2];
    Bf hc_down[2], hc_up[2], hc_inj[2];
    Dev g_inp, s_gate_inp;
    Q8 sh_gate, sh_up, sh_down;
};

struct PleSrc {
    uint8_t* base = nullptr;
    size_t sz = 0;
    uint64_t data_off = 0, row_off = 0, rows = 0;
    int row_bytes = 0;
    float scale = 1.f;
    bool open(const char* path) {
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat stt;
        if (fstat(fd, &stt) != 0) { ::close(fd); return false; }
        sz = (size_t) stt.st_size;
        base = (uint8_t*) mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) { base = nullptr; return false; }
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
        return rows > 0 && row_bytes == PLE_HEAD;
    }
    void row(uint64_t idx, float* out) const {
        const uint8_t* p = base + data_off + row_off + idx * (uint64_t) row_bytes;
        for (int i = 0; i < PLE_HEAD; ++i) {
            const uint8_t v = p[i];
            const int sg = (v >> 7) & 1, e = (v >> 3) & 0xF, mm = v & 7;
            float f;
            if (e == 0) f = std::ldexp((float) mm, -9);
            else if (e == 15 && mm == 7) f = NAN;
            else f = std::ldexp(1.f + (float) mm / 8.f, e - 7);
            out[i] = (sg ? -f : f) * scale;
        }
    }
};

// Per-GPU Stage State & Execution Context
struct StageGpu {
    int dev = 0;
    int l_begin = 0, l_end = 24;
    int ctx_max = 16384;
    int n_qsa = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t exec_body = nullptr;
    cudaGraphExec_t exec_full = nullptr;

    std::vector<LayerW> L;
    std::vector<Dev> conv, ssm;
    Dev kcache, vcache, qsa_sc;
    Dev R;

    // per-stage scratch buffers
    Dev xn, lo, gg, gated, mixed, mixed2, inject, block;
    Dev qkv, convout, z, alpha, beta, decay, bt, o, on;
    Dev qg, kk, vv, attn;
    Dev slog, sids, swts, sff, sup, sshared, sgate, tmp_ff;
    Dev ff3, parts, comb, ffact, exsc;
    signed char *xq8 = nullptr, *xq4 = nullptr, *eq16 = nullptr, *eq32 = nullptr, *dq16 = nullptr, *dq32 = nullptr;
    float *xs8 = nullptr, *xs4 = nullptr, *es16 = nullptr, *es32 = nullptr, *ds16 = nullptr, *ds32 = nullptr;

    // PLE (only on Stage 0)
    bool has_ple = false;
    Bf ple_wk, ple_wv;
    Dev ple_nk, ple_nq, ple_nc, ple_cv, ple_hist, d_ple_emb;
    Dev ple_q, ple_k, ple_v, ple_g, ple_gat, ple_n;

    // Output Head (only on last Stage)
    bool has_head = false;
    Dev headnorm, hlogits;
    Bf head_down, head_up;
    Q8 headq;

    // Direct O(1) LUT + Hybrid VRAM + Pinned-Host-Pack Expert Cache
    int cap = 0;             // total VRAM slots on this GPU
    int adapt_cap = 256;     // ring buffer at [cap - adapt_cap, cap) for adaptive promotion
    int static_cap = 0;      // [0, static_cap) permanently holds top profile experts
    int ring_head = 0;
    uint8_t* xblobs = nullptr;
    uint8_t* d_pack = nullptr;
    int* d_lut = nullptr;
    std::vector<int> h_lut;
    std::vector<int> ring_owner;
    std::vector<uint16_t> miss_cnt;
    int *dctx = nullptr, *dpos = nullptr, *dmisses = nullptr, *dslots = nullptr, *mkey = nullptr;
    std::vector<int> h_mkey;
    long long total_lookups = 0, total_vram_hits = 0, total_promotions = 0;

    void dense_gemv(const Q8& W, const float* x, float* y, int rows, int cols) {
        quant_act<<<((cols + 15) / 16 + 127) / 128, 128, 0, stream>>>(x, xq8, xs8, xq4, xs4, cols);
        gemv_q8p<<<(rows + 7) / 8, 256, 0, stream>>>(W.p, W.s, xq8, xs8, y, rows, cols);
    }

    void gr_read(const Dev& gn, const Bf& gd, const Bf& gu, const Bf& gi, Dev& mx, Dev& inj) {
        group_norm<<<HC, 256, 0, stream>>>(R.p, gn.p, xn.p, NE, EPS);
        CHECK(cudaMemsetAsync(lo.p, 0, HCLR * 4, stream));
        gemv_bf16_split<<<(HCLR * 8 + 7) / 8, 256, 0, stream>>>(gd.p, xn.p, lo.p, HCLR, HCD, 8);
        silu_inplace<<<(HCLR + 255) / 256, 256, 0, stream>>>(lo.p, HCLR, 1.f / (float) HC);
        gemv_bf16<<<(HCD + 7) / 8, 256, 0, stream>>>(gu.p, lo.p, gg.p, HCD, HCLR);
        gr_gate<<<(HCD + 255) / 256, 256, 0, stream>>>(xn.p, gg.p, gated.p, HCD);
        gr_mean<<<(NE + 255) / 256, 256, 0, stream>>>(gated.p, mx.p, NE, HC, (float) HC);
        CHECK(cudaMemsetAsync(inj.p, 0, HC * 4, stream));
        gemv_bf16_split<<<(HC * 16 + 7) / 8, 256, 0, stream>>>(gi.p, xn.p, inj.p, HC, HCD, 16);
    }

    void record_forward(const int* h_ctx, const int* h_pos, const float* h_emb, const float* h_ple, bool run_head) {
        CHECK(cudaSetDevice(dev));
        CHECK(cudaMemsetAsync(dmisses, 0, 4, stream));
        CHECK(cudaMemcpyAsync(dctx, h_ctx, 4, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(dpos, h_pos, 4, cudaMemcpyHostToDevice, stream));
        if (l_begin == 0) {
            for (int c = 0; c < HC; ++c)
                CHECK(cudaMemcpyAsync(R.p + (size_t) c * NE, h_emb, NE * 4, cudaMemcpyHostToDevice, stream));
        }
        if (has_ple) {
            CHECK(cudaMemcpyAsync(d_ple_emb.p, h_ple, (size_t) NE * 4, cudaMemcpyHostToDevice, stream));
        }

        for (int l = l_begin; l < l_end; ++l) {
            LayerW& w = L[l];
            if (l == 1 && has_ple) {
                ple_group<<<HC, 256, 0, stream>>>(R.p, ple_nq.p, ple_q.p, NE, HC, EPS);
                gemv_bf16<<<(HCD + 7) / 8, 256, 0, stream>>>(ple_wk.p, d_ple_emb.p, ple_k.p, HCD, NE);
                ple_group<<<HC, 256, 0, stream>>>(ple_k.p, ple_nk.p, ple_k.p, NE, HC, EPS);
                gemv_bf16<<<(NE + 7) / 8, 256, 0, stream>>>(ple_wv.p, d_ple_emb.p, ple_v.p, NE, NE);
                ple_gate<<<HC, 128, 0, stream>>>(ple_k.p, ple_q.p, ple_g.p, HC, NE);
                ple_gv<<<(HCD + 255) / 256, 256, 0, stream>>>(ple_v.p, ple_g.p, ple_gat.p, HCD, NE);
                ple_group<<<HC, 256, 0, stream>>>(ple_gat.p, ple_nc.p, ple_n.p, NE, HC, EPS);
                ple_conv<<<(HCD + 255) / 256, 256, 0, stream>>>(ple_n.p, ple_cv.p, ple_hist.p,
                                                                 ple_gat.p, R.p, HCD, PLE_NG, PLE_KERN);
            }

            gr_read(w.hc_norm[0], w.hc_down[0], w.hc_up[0], w.hc_inj[0], mixed, inject);
            CHECK(cudaMemsetAsync(block.p, 0, (size_t) NE * 4, stream));

            if (!w.qsa) {
                dense_gemv(w.qkv, mixed.p, qkv.p, QKV, NE);
                gdn_conv<<<(QKV + 255) / 256, 256, 0, stream>>>(conv[l].p, qkv.p, w.conv1d.p, convout.p, QKV, 4);
                l2_norm<<<GH, 128, 0, stream>>>(convout.p, ST, EPS);
                l2_norm<<<GH, 128, 0, stream>>>(convout.p + GQ, ST, EPS);
                dense_gemv(w.wgate, mixed.p, z.p, GV, NE);
                gemv_f32<<<(VH + 7) / 8, 256, 0, stream>>>(w.alpha.p, mixed.p, alpha.p, VH, NE);
                gemv_f32<<<(VH + 7) / 8, 256, 0, stream>>>(w.beta.p, mixed.p, beta.p, VH, NE);
                gdn_gate<<<1, 64, 0, stream>>>(alpha.p, w.dtb.p, w.ssm_a.p, beta.p, decay.p, bt.p, VH);
                dim3 gs((ST + 127) / 128, VH);
                gdn_step<<<gs, 128, (size_t) (2 * ST + 2) * 4, stream>>>(ssm[l].p, convout.p, convout.p + GQ,
                                                                 convout.p + 2 * GQ, decay.p, bt.p, o.p, ST, GH, VH);
                gdn_out_norm<<<VH, 128, 0, stream>>>(o.p, z.p, w.ssm_norm.p, on.p, ST, EPS);
                dense_gemv(w.wout, on.p, block.p, NE, GV);
            } else {
                dense_gemv(w.wq, mixed.p, qg.p, 2 * QD, NE);
                dense_gemv(w.wk, mixed.p, kk.p, KVD, NE);
                dense_gemv(w.wv, mixed.p, vv.p, KVD, NE);
                head_norm<<<QH, 128, 0, stream>>>(qg.p, w.q_norm.p, HD, 2 * HD, EPS);
                head_norm<<<KVH, 128, 0, stream>>>(kk.p, w.k_norm.p, HD, HD, EPS);
                rope_apply<<<QH, 32, 0, stream>>>(qg.p, QH, dpos, 2 * HD, NROT, ROPE_THETA);
                rope_apply<<<KVH, 32, 0, stream>>>(kk.p, KVH, dpos, HD, NROT, ROPE_THETA);
                float* kc_l = kcache.p + (size_t) w.qsa_local * ctx_max * KVD;
                float* vc_l = vcache.p + (size_t) w.qsa_local * ctx_max * KVD;
                kv_append<<<(KVD + 127) / 128, 128, 0, stream>>>(kc_l, vc_l, kk.p, vv.p, dctx, KVD);
                qsa_attend_fast<<<QH, HD, 0, stream>>>(qg.p, kc_l, vc_l, qsa_sc.p, ctx_max,
                                                       attn.p, dctx, QH, KVH, HD, 2 * HD, HD);
                dense_gemv(w.wo, attn.p, block.p, NE, QD);
            }
            gr_write_k<<<80, 256, HC * 4, stream>>>(R.p, block.p, inject.p, R.p, NE, HC, 2.f, (float) HC);

            // FFN half
            gr_read(w.hc_norm[1], w.hc_down[1], w.hc_up[1], w.hc_inj[1], mixed2, inject);
            gemv_f32<<<(NEXP + 7) / 8, 256, 0, stream>>>(w.g_inp.p, mixed2.p, slog.p, NEXP, NE);
            router_topk<<<1, 256, (size_t) (NEXP * 12 + 4 * NEXP), stream>>>(slog.p, NEXP, TOPK,
                                                                    (int*) sids.p, swts.p);
            dense_gemv(w.sh_gate, mixed2.p, sff.p, NFF, NE);
            dense_gemv(w.sh_up, mixed2.p, sup.p, NFF, NE);
            silu_mul<<<(NFF + 255) / 256, 256, 0, stream>>>(sff.p, sup.p, tmp_ff.p, NFF);
            dense_gemv(w.sh_down, tmp_ff.p, sshared.p, NE, NFF);
            dot1<<<1, 256, 0, stream>>>(mixed2.p, w.s_gate_inp.p, sgate.p, NE);
            sigmoid_scale_dev<<<(NE + 255) / 256, 256, 0, stream>>>(sshared.p, sgate.p, NE);

            map_slots_lut<<<1, 32, 0, stream>>>((const int*) sids.p, TOPK, d_lut, l, dslots, dmisses, mkey);
            exp_scales<<<1, 32, 0, stream>>>(xblobs, d_pack, (const int*) dslots, TOPK, exsc.p);
            quant_act<<<((NE + 15) / 16 + 127) / 128, 128, 0, stream>>>(mixed2.p, eq32, es32, eq16, es16, NE);
            exp_gu<<<(TOPK * NFF + 7) / 8, 256, 0, stream>>>(xblobs, d_pack, (const int*) dslots, TOPK,
                      eq16, es16, ff3.p, exsc.p, NFF, NE, GU_ROW);
            exp_gu_epi<<<(TOPK * NFF + 127) / 128, 128, 0, stream>>>(ff3.p, ffact.p, TOPK * NFF);
            quant_act<<<((TOPK * NFF + 15) / 16 + 127) / 128, 128, 0, stream>>>(
                ffact.p, dq32, ds32, dq16, ds16, TOPK * NFF);
            exp_down<<<(TOPK * NE + 7) / 8, 256, 0, stream>>>(xblobs, d_pack, (const int*) dslots, TOPK,
                      dq16, ds16, parts.p, exsc.p, NE, NFF, D_ROW);
            moe_combine<<<(NE + 255) / 256, 256, 0, stream>>>(parts.p, swts.p, sshared.p, comb.p, NE, TOPK);
            gr_write_k<<<80, 256, HC * 4, stream>>>(R.p, comb.p, inject.p, R.p, NE, HC, 2.f, (float) HC);
        }

        if (has_head && run_head) {
            group_norm<<<HC, 256, 0, stream>>>(R.p, headnorm.p, xn.p, NE, EPS);
            gemv_bf16<<<(HCLR + 7) / 8, 256, 0, stream>>>(head_down.p, xn.p, lo.p, HCLR, HCD);
            silu_inplace<<<(HCLR + 255) / 256, 256, 0, stream>>>(lo.p, HCLR, 1.f / (float) HC);
            gemv_bf16<<<(HCD + 7) / 8, 256, 0, stream>>>(head_up.p, lo.p, gg.p, HCD, HCLR);
            gr_gate<<<(HCD + 255) / 256, 256, 0, stream>>>(xn.p, gg.p, gated.p, HCD);
            gr_mean<<<(NE + 255) / 256, 256, 0, stream>>>(gated.p, mixed.p, NE, HC, (float) HC);
            dense_gemv(headq, mixed.p, hlogits.p, NVOCAB, NE);
        }
    }

    void promote_cold(const uint8_t* host_pack_raw, bool do_promote) {
        CHECK(cudaSetDevice(dev));
        int n = 0;
        CHECK(cudaMemcpyAsync(&n, dmisses, 4, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        const int stage_exp = (l_end - l_begin) * TOPK;
        total_lookups += stage_exp;
        const int hits = stage_exp - std::min(n, stage_exp);
        total_vram_hits += hits;
        if (!do_promote || n <= 0 || adapt_cap <= 0) return;
        const int take = std::min(n, 512);
        CHECK(cudaMemcpyAsync(h_mkey.data(), mkey, (size_t) take * 4, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        int promoted = 0;
        for (int i = 0; i < take && promoted < 32; ++i) {
            const int gkey = h_mkey[i];
            if (gkey < 0 || gkey >= 48 * NEXP || h_lut[gkey] >= 0) continue;
            const int r = ring_head++ % adapt_cap;
            const int vslot = static_cap + r;
            const int old_k = ring_owner[r];
            if (old_k >= 0) {
                h_lut[old_k] = -1;
                CHECK(cudaMemcpyAsync(d_lut + old_k, &h_lut[old_k], 4, cudaMemcpyHostToDevice, stream));
            }
            CHECK(cudaMemcpyAsync(xblobs + (size_t) vslot * PER,
                                  host_pack_raw + (size_t) gkey * PER,
                                  PER, cudaMemcpyHostToDevice, stream));
            ring_owner[r] = gkey;
            h_lut[gkey] = vslot;
            CHECK(cudaMemcpyAsync(d_lut + gkey, &h_lut[gkey], 4, cudaMemcpyHostToDevice, stream));
            ++promoted;
            ++total_promotions;
        }
        if (promoted > 0) CHECK(cudaStreamSynchronize(stream));
    }

    void reset_state() {
        CHECK(cudaSetDevice(dev));
        for (int l = l_begin; l < l_end; ++l) {
            if (conv[l].p) CHECK(cudaMemsetAsync(conv[l].p, 0, (size_t) QKV * 3 * 4, stream));
            if (ssm[l].p) CHECK(cudaMemsetAsync(ssm[l].p, 0, (size_t) VH * ST * ST * 4, stream));
        }
        if (has_ple && ple_hist.p)
            CHECK(cudaMemsetAsync(ple_hist.p, 0, (size_t) HCD * PLE_HIST * 4, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
};
