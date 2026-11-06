// wzy_batch_engine.cu -- Concurrent multi-session engine for 2x sm_75.
//
// What this engine is:
//   * The v2 layer executor (StageGpu::record_forward) is used UNCHANGED. A "session" is a set of
//     pointers (KV ring, GDN state, indexer pool, cold-expert mailbox, scratch, logits) swapped in by
//     session_bind(). Each session therefore runs exactly the single-stream arithmetic, bit for bit.
//   * Up to WZY_MAX_SESS sessions are resident at once and decode CONCURRENTLY on private streams.
//     The GPU is latency/barrier bound (~6x off its bandwidth ceiling in decode), so overlapping
//     several sessions fills the SMs that one session leaves idle, and one session's CPU cold-expert
//     barrier overlaps another session's GPU compute.
//   * Cold experts are computed by the shared AVX2 pool; promotion into VRAM is deferred to the end of
//     the round so a session never evicts an expert another session is reading.
//   * Multi-turn prefix reuse is by full-prefix match on an idle slot: no park/restore, no 20 ms copy.
//
// What this engine deliberately is NOT:
//   * No MTP and no speculative window. A verify window needs per-row GDN state snapshots
//     (18 layers x 8 rows x 3 MiB = 453 MB per stage per session); at B=2..4 that does not fit next to
//     the expert cache. Batched decode therefore runs one row per step. Single-stream users keep the
//     v2 binary (wzy_pp2_engine) with MTP at 80-90 tok/s.
//   * ctx_max <= KV_VRAM_RING (32768). Beyond that the ring would overwrite live cells without the
//     256K host mirror; long-context requests belong on the v2 binary.
//
// Protocol (stdin, one JSON per line):
//   {"req":7,"ids":[...],"max_tokens":256,"temperature":0.3,...,"stop":[...]}
//   CANCEL 7
//   PING / QUIT
// stdout:
//   PREFILL {"req":7,...}
//   TOK 7 1234
//   DONE {"req":7,...}
#include "wzy_mtp.cuh"
#include "wzy_prefill_tc.cuh"
#include "wzy_batch.cuh"

#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

static volatile sig_atomic_t g_cancel_req = 0;
static void on_sigusr1_cancel(int) { g_cancel_req = 1; }

// The stdin reader answers PING while the scheduler emits TOK lines. Without a lock the two can
// interleave inside a line and break the parent's parser, so every protocol line goes through `say`.
static std::mutex g_out_mu;
static void say(const char* fmt, ...) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

// ---------------------------------------------------------------- engine-wide state
static std::vector<StageGpu> stages;
static std::vector<PrefillStageCtx> pf_ctx;
static bool pp2 = false;
static int nstages = 1;
static int ctx_max = KV_VRAM_RING;
static int prefill_chunk = 25600;
static int split_pf_chunks = 2;
static bool explicit_pf_chunk = false;
static uint8_t* host_pack = nullptr;
static CpuMoePool cpu_pool;
static PleSrc plesrc;
static cudaEvent_t ev_s0_done = nullptr;
static cudaEvent_t ev_s1_rcopied = nullptr;
static Session sess[WZY_MAX_SESS];
static int nsess = 1;
static std::mt19937 rng(std::random_device{}());

// Token embedding table (mmap'd; never freed).
static const uint16_t* g_bf_emb = nullptr;
static const uint8_t* g_q_emb = nullptr;
static int g_emb_nb = 0;

static void stage_one_emb(float* dst, int id) {
    if (id < 0 || id >= NVOCAB) id = 0;
    if (g_bf_emb != nullptr) {
        const uint16_t* r = g_bf_emb + (size_t) id * NE;
        for (int i = 0; i < NE; ++i) dst[i] = bf2f(r[i]);
        return;
    }
    const uint8_t* row = g_q_emb + (size_t) id * g_emb_nb * 34;
    for (int b = 0; b < g_emb_nb; ++b) {
        const float d = h2f(*(const uint16_t*) (row + b * 34));
        const int8_t* qp = (const int8_t*) (row + b * 34 + 2);
        for (int i = 0; i < 32; ++i) dst[b * 32 + i] = d * qp[i];
    }
}

// ---------------------------------------------------------------- 16-head 3-gram PLE
// Same hashing as the v2 host path; `advance` updates the session's 2-token history.
static void ple_window(Session& S, const int* toks, int T, bool advance, float* buf) {
    static const uint64_t ple_mult[3] = {23703573157769ull, 20109073645365ull, 8052911324071ull};
    static const uint64_t ple_vocab[PLE_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    static const uint64_t ple_off[PLE_HEADS] = {
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

    int p0 = S.ple_prev[0], p1 = S.ple_prev[1];
    uint32_t r_win[MAX_W][PLE_HEADS];
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
        p0 = p1;
        p1 = cur;
    }
    for (int t = 0; t < T; ++t) {
        float* dst = buf + (size_t) t * NE;
        for (int h = 0; h < PLE_HEADS; ++h) plesrc.row(r_win[t][h], dst + (size_t) h * PLE_HEAD);
    }
    if (advance) {
        if (T >= 2) {
            S.ple_prev[0] = toks[T - 2];
            S.ple_prev[1] = toks[T - 1];
        } else if (T == 1) {
            S.ple_prev[0] = S.ple_prev[1];
            S.ple_prev[1] = toks[0];
        }
    }
}

// Vision rows: overwrite the embeddings of prompt tokens that belong to an image run.
static void inject_img_emb(Session& S, float* dst, int p_start, int B) {
    if (S.img_start.empty()) return;
    long off = 0;
    for (size_t k = 0; k < S.img_start.size(); ++k) {
        const int s = S.img_start[k], c = S.img_count[k];
        const int lo = std::max(p_start, s);
        const int hi = std::min(p_start + B, s + c);
        for (int gi = lo; gi < hi; ++gi) {
            std::memcpy(dst + (size_t) (gi - p_start) * NE,
                        S.img_emb.data() + (size_t) (off + (gi - s)) * NE,
                        (size_t) NE * sizeof(float));
        }
        off += c;
    }
}

static bool load_img_emb(Session& S, const std::string& path, const std::vector<int>& starts,
                         const std::vector<int>& counts) {
    S.img_emb.clear();
    S.img_start.clear();
    S.img_count.clear();
    if (path.empty() || starts.empty() || starts.size() != counts.size()) return true;
    long total = 0;
    for (int c : counts) {
        if (c <= 0) return false;
        total += c;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[wzy-batch] cannot open %s\n", path.c_str());
        return false;
    }
    int32_t hdr[5] = {0, 0, 0, 0, 0};
    f.read((char*) hdr, sizeof hdr);
    if (hdr[0] != 0x31455653 || hdr[1] < total || hdr[4] != NE) {
        std::fprintf(stderr, "[wzy-batch] bad SVE header magic=%08x rows=%d dim=%d\n", hdr[0], hdr[1], hdr[4]);
        return false;
    }
    S.img_emb.resize((size_t) total * NE);
    f.read((char*) S.img_emb.data(), (std::streamsize) ((size_t) total * NE * sizeof(float)));
    if (!f) {
        S.img_emb.clear();
        return false;
    }
    S.img_start = starts;
    S.img_count = counts;
    return true;
}

// Embedding + PLE for a prefill chunk, spread across 16 cores (same as v2).
static void stage_emb_ple_pf(Session& S, const int* all_ids, int p_start, int B, float* dst_emb, float* dst_ple) {
    static const uint64_t ple_mult[3] = {23703573157769ull, 20109073645365ull, 8052911324071ull};
    static const uint64_t ple_vocab[PLE_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    static const uint64_t ple_off[PLE_HEADS] = {
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

    const int prev_m2 = (p_start >= 2) ? all_ids[p_start - 2]
                       : ((p_start == 1) ? S.ple_prev[1] : S.ple_prev[0]);
    const int prev_m1 = (p_start >= 1) ? all_ids[p_start - 1] : S.ple_prev[1];

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
        S.ple_prev[0] = all_ids[p_start + B - 2];
        S.ple_prev[1] = all_ids[p_start + B - 1];
    } else if (B == 1) {
        S.ple_prev[0] = S.ple_prev[1];
        S.ple_prev[1] = all_ids[p_start];
    }
}

// ---------------------------------------------------------------- session lifecycle
static void session_reset(Session& S) {
    for (int si = 0; si < nstages; ++si) {
        StageGpu& stg = stages[si];
        session_bind(stg, S, si);
        CHECK(cudaSetDevice(stg.dev));
        for (int l = stg.l_begin; l < stg.l_end; ++l) {
            if (!stg.L[l].qsa) {
                CHECK(cudaMemsetAsync(stg.conv[l].p, 0, (size_t) QKV * 3 * sizeof(float), stg.stream));
                CHECK(cudaMemsetAsync(stg.ssm[l].p, 0, (size_t) VH * ST * ST * sizeof(float), stg.stream));
            }
        }
        if (stg.has_ple) {
            CHECK(cudaMemsetAsync(stg.ple_hist.p, 0, (size_t) HCD * PLE_HIST * sizeof(float), stg.stream));
        }
        CHECK(cudaMemsetAsync(stg.vk_q, 0, (size_t) stg.n_qsa * KV_VRAM_RING * KVD, stg.stream));
        CHECK(cudaMemsetAsync(stg.vv_q, 0, (size_t) stg.n_qsa * KV_VRAM_RING * KVD, stg.stream));
        CHECK(cudaMemsetAsync(stg.vk_s, 0, (size_t) stg.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * sizeof(unsigned short), stg.stream));
        CHECK(cudaMemsetAsync(stg.vv_s, 0, (size_t) stg.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * sizeof(unsigned short), stg.stream));
        CHECK(cudaMemsetAsync(stg.idx_tail.p, 0, stg.idx_tail.n * sizeof(float), stg.stream));
        CHECK(cudaMemsetAsync(stg.idx_dead.p, 0, stg.idx_dead.n * sizeof(float), stg.stream));
        CHECK(cudaMemsetAsync(stg.idx_pooled.p, 0, stg.idx_pooled.n * sizeof(float), stg.stream));
        CHECK(cudaMemsetAsync(stg.dpos, 0, sizeof(int), stg.stream));
        CHECK(cudaStreamSynchronize(stg.stream));
    }
    S.ple_prev[0] = -1;
    S.ple_prev[1] = -1;
}

// One synchronous row: feed `tok` at `pos`, run both stages, service cold experts, sample nothing.
// Used for the last prompt token and for the prefill short path.
static void run_one_row(Session& S, int tok, int pos, bool need_logits) {
    *S.h_pos = pos;
    stage_one_emb(S.h_emb, tok);
    // The final prompt token can itself be an image pad; v2 injects it in run_verify_window too.
    inject_img_emb(S, S.h_emb, pos, 1);
    ple_window(S, &tok, 1, true, S.h_ple);

    session_bind(stages[0], S, 0);
    S.base_seq[0] = stages[0].cold_seq;
    stages[0].record_forward(S.h_pos, S.h_emb, S.h_ple, 1, !pp2 && need_logits, false,
                             host_pack, &cpu_pool, S.want_top16);
    // Stage 1 must not start before stage 0's R row is final: every pp2 path waits on this event.
    CHECK(cudaEventRecord(S.ev[0], S.st[0]));

    if (pp2) {
        session_bind(stages[1], S, 1);
        CHECK(cudaSetDevice(stages[1].dev));
        CHECK(cudaStreamWaitEvent(S.st[1], S.ev[0], 0));
        CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev, S.R[0].p, stages[0].dev,
                                  (size_t) HCD * sizeof(float), S.st[1]));
        S.base_seq[1] = stages[1].cold_seq;
        stages[1].record_forward(S.h_pos, S.h_emb, S.h_ple, 1, true, false,
                                 host_pack, &cpu_pool, S.want_top16);
        if (need_logits && !S.fast_greedy && !S.want_top16) {
            CHECK(cudaMemcpyAsync(S.h_logits, stages[1].hlogits.p, (size_t) NVOCAB * sizeof(float),
                                  cudaMemcpyDeviceToHost, S.st[1]));
        }
        CHECK(cudaEventRecord(S.ev[1], S.st[1]));
    } else {
        if (need_logits && !S.fast_greedy && !S.want_top16) {
            CHECK(cudaMemcpyAsync(S.h_logits, stages[0].hlogits.p, (size_t) NVOCAB * sizeof(float),
                                  cudaMemcpyDeviceToHost, S.st[0]));
        }
        CHECK(cudaEventRecord(S.ev[0], S.st[0]));
    }

    for (int si = 0; si < nstages; ++si) {
        session_bind(stages[si], S, si);
        stages[si].service_cold_layers(host_pack, &cpu_pool, S.base_seq[si], 1, false);
    }
    CHECK(cudaSetDevice(stages.back().dev));
    CHECK(cudaEventSynchronize(S.ev[pp2 ? 1 : 0]));
    for (int si = 0; si < nstages; ++si) {
        session_bind(stages[si], S, si);
        stages[si].promote_cold(host_pack, 1, true);
    }
}

// Short-prompt prefill step: up to MAX_W rows through the ordinary layer path (no snapshots).
static void run_prefill_step(Session& S, const int* toks, int T, int pos) {
    *S.h_pos = pos;
    for (int t = 0; t < T; ++t) stage_one_emb(S.h_emb + (size_t) t * NE, toks[t]);
    inject_img_emb(S, S.h_emb, pos, T);
    ple_window(S, toks, T, true, S.h_ple);

    session_bind(stages[0], S, 0);
    S.base_seq[0] = stages[0].cold_seq;
    stages[0].record_forward(S.h_pos, S.h_emb, S.h_ple, T, false, false, host_pack, &cpu_pool, S.want_top16);
    CHECK(cudaEventRecord(S.ev[0], S.st[0]));

    if (pp2) {
        session_bind(stages[1], S, 1);
        CHECK(cudaSetDevice(stages[1].dev));
        CHECK(cudaStreamWaitEvent(S.st[1], S.ev[0], 0));
        CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev, S.R[0].p, stages[0].dev,
                                  (size_t) T * HCD * sizeof(float), S.st[1]));
        S.base_seq[1] = stages[1].cold_seq;
        stages[1].record_forward(S.h_pos, S.h_emb, S.h_ple, T, false, false,
                                 host_pack, &cpu_pool, S.want_top16);
        CHECK(cudaEventRecord(S.ev[1], S.st[1]));
    }

    for (int si = 0; si < nstages; ++si) {
        session_bind(stages[si], S, si);
        stages[si].service_cold_layers(host_pack, &cpu_pool, S.base_seq[si], T, false);
    }
    CHECK(cudaSetDevice(stages.back().dev));
    CHECK(cudaEventSynchronize(S.ev[pp2 ? 1 : 0]));
    for (int si = 0; si < nstages; ++si) {
        session_bind(stages[si], S, si);
        stages[si].promote_cold(host_pack, T, true);
    }
}

// Prefill [p_begin, p_end) with the Tensor-Core chunked path, dual-GPU pipeline overlapped.
static void run_prefill_range(Session& S, const int* all_ids, int nprompt, int p_begin, int p_end) {
    (void) nprompt;
    const int total_pf = p_end - p_begin;
    if (total_pf <= 0) return;

    if (total_pf <= 192) {
        for (int p = p_begin; p < p_end;) {
            const int step_w = std::min((int) MAX_W, p_end - p);
            run_prefill_step(S, all_ids + p, step_w, p);
            p += step_w;
        }
        return;
    }

    cpu_pool.set_sleep(true);
    struct ChunkSpan { int p; int B; };
    std::vector<ChunkSpan> chunks;
    int eff_chunk = prefill_chunk;
    if (pp2 && !explicit_pf_chunk && split_pf_chunks > 1 && total_pf >= 16384) {
        const int want_n = std::max(split_pf_chunks, (total_pf + prefill_chunk - 1) / prefill_chunk);
        eff_chunk = std::min(prefill_chunk, ((total_pf + want_n - 1) / want_n + 63) & ~63);
    }
    for (int p = p_begin; p < p_end; p += eff_chunk) chunks.push_back({p, std::min(eff_chunk, p_end - p)});

    std::thread s1_thread;
    std::atomic<bool> s1_p2p_queued{true};
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const int p = chunks[ci].p, B = chunks[ci].B;
        if (ci > 0) {
            CHECK(cudaSetDevice(stages[0].dev));
            CHECK(cudaEventSynchronize(pf_ctx[0].ev_in_copied));
        }
        stage_emb_ple_pf(S, all_ids, p, B, pf_ctx[0].h_emb_pf, pf_ctx[0].h_ple_pf);
        inject_img_emb(S, pf_ctx[0].h_emb_pf, p, B);

        CHECK(cudaSetDevice(stages[0].dev));
        if (ci > 0 && pp2) {
            CHECK(cudaEventSynchronize(ev_s0_done));
            while (!s1_p2p_queued.load(std::memory_order_acquire)) _mm_pause();
            CHECK(cudaStreamWaitEvent(stages[0].stream, ev_s1_rcopied, 0));
        }
        session_bind(stages[0], S, 0);
        float* R_s0 = pf_ctx[0].run_stage_prefill(stages[0], p, B, host_pack, nullptr);

        if (!pp2) {
            CHECK(cudaStreamSynchronize(stages[0].stream));
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
            session_bind(stages[1], S, 1);
            pf_ctx[1].run_stage_prefill(stages[1], p, B, host_pack, R_s0, ev_s1_rcopied, &s1_p2p_queued);
            CHECK(cudaStreamSynchronize(stages[1].stream));
        });
    }
    if (s1_thread.joinable()) s1_thread.join();
    CHECK(cudaSetDevice(stages[0].dev));
    CHECK(cudaStreamSynchronize(stages[0].stream));
    for (int si = 0; si < nstages; ++si) pf_ctx[si].sync_stats(stages[si]);
    cpu_pool.set_sleep(false);
}

// ---------------------------------------------------------------- sampling / emission
static bool is_stop_tok(const Session& S, int tok) {
    for (int s : S.cfg.stops) if (tok == s) return true;
    return false;
}

static int sample_next(Session& S) {
    int tok = 0;
    if (S.fast_greedy) {
        tok = S.h_greedy[0];
    } else if (S.want_top16) {
        verify_or_resample_top16(S.h_top16v, S.h_top16i, S.cfg, S.tokens, (size_t) S.prompt_len,
                                 -1, tok, rng);
    } else {
        verify_or_resample(S.h_logits, NVOCAB, S.cfg, S.tokens, -1, tok, rng);
    }
    return tok;
}

static void emit_tok(Session& S, int tok) {
    say("TOK %d %d\n", S.id, tok);
    ++S.generated;
}

static void emit_done(Session& S, const char* finish, double prompt_ms) {
    long long lk = 0, vh = 0, pr = 0;
    for (const auto& stg : stages) {
        lk += stg.total_lookups;
        vh += stg.total_vram_hits;
        pr += stg.total_promotions;
    }
    const double now_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const double decode_ms = (S.t_first > 0.0) ? (now_s - S.t_first) * 1000.0 : 0.0;
    say("DONE {\"req\":%d,\"generated\":%d,\"prompt_tokens\":%d,\"reused\":%d,"
        "\"prompt_ms\":%.2f,\"decode_ms\":%.2f,\"draft_n\":0,\"draft_accepted\":0,"
        "\"vram_hit\":%.4f,\"promoted\":%lld,\"restored_park\":0,\"parked_tokens\":0,"
        "\"active_tokens\":%d,\"finish\":\"%s\"}\n",
        S.id, S.generated, S.prompt_len, S.reused_tokens, prompt_ms, decode_ms,
        lk > 0 ? (double) vh / (double) lk : 0.0, pr, (int) S.tokens.size(), finish);
}

// ---------------------------------------------------------------- prefill / admission
struct PendingReq {
    int req = 0;
    std::vector<int> ids;
    GenCfg cfg;
    std::string emb_file;
    std::vector<int> emb_starts, emb_counts;
};

static void prefill_session(Session& S, const PendingReq& r) {
    const int nprompt = (int) r.ids.size();
    const auto t0 = std::chrono::steady_clock::now();

    // Full-prefix reuse on an idle slot: keep the resident state, prefill only the suffix.
    int reused = 0;
    if (!S.tokens.empty() && (int) S.tokens.size() <= nprompt - 1) {
        bool prefix = true;
        for (size_t k = 0; k < S.tokens.size(); ++k) {
            if (S.tokens[k] != r.ids[k]) { prefix = false; break; }
        }
        if (prefix) reused = (int) S.tokens.size();
    }
    if (reused == 0) {
        session_reset(S);
        S.tokens.clear();
        S.ple_prev[0] = -1;
        S.ple_prev[1] = -1;
    } else {
        S.ple_prev[0] = (reused >= 2) ? r.ids[reused - 2] : -1;
        S.ple_prev[1] = r.ids[reused - 1];
    }

    S.id = r.req;
    S.cfg = r.cfg;
    S.max_new = r.cfg.max_new;
    S.generated = 0;
    S.finished = false;
    S.cancelled = false;
    S.active = false;
    S.t_first = 0.0;
    S.tokens.assign(r.ids.begin(), r.ids.end());
    S.prompt_len = nprompt;
    S.reused_tokens = reused;
    S.pos = nprompt;

    const bool no_rep = (std::fabs(S.cfg.rep_pen - 1.0f) < REP_PEN_EPS);
    S.fast_greedy = no_rep && (S.cfg.temp <= 1e-5f);
    S.want_top16 = no_rep && !S.fast_greedy && (S.cfg.top_k > 0 && S.cfg.top_k <= 16);

    load_img_emb(S, r.emb_file, r.emb_starts, r.emb_counts);

    run_prefill_range(S, r.ids.data(), nprompt, reused, nprompt - 1);
    if (nprompt - 1 - reused > 192) {
        for (int si = 0; si < nstages; ++si) pf_ctx[si].restore_borrowed(stages[si], host_pack);
        for (int si = 0; si < nstages; ++si) {
            CHECK(cudaSetDevice(stages[si].dev));
            CHECK(cudaStreamSynchronize(stages[si].stream));
        }
    }

    // Feed the last prompt token and sample the first generated token.
    run_one_row(S, r.ids[nprompt - 1], nprompt - 1, true);

    const double pf_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count();
    const int pf_read = nprompt - reused;
    say("PREFILL {\"req\":%d,\"prompt_tokens\":%d,\"reused\":%d,\"read\":%d,"
        "\"prompt_ms\":%.2f,\"prompt_tps\":%.1f,\"restored_park\":0,\"parked_tokens\":0}\n",
        S.id, nprompt, reused, pf_read, pf_ms,
        pf_ms > 0 ? 1000.0 * pf_read / pf_ms : 0.0);

    S.img_emb.clear();
    S.img_start.clear();
    S.img_count.clear();

    int tok = sample_next(S);
    if (is_stop_tok(S, tok)) {
        emit_done(S, "stop", pf_ms);
        S.active = false;
        S.finished = true;
        return;
    }
    S.t_first = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
    S.cur_tok = tok;
    S.active = true;
    emit_tok(S, tok);
    if (S.generated >= S.max_new) {
        emit_done(S, "length", pf_ms);
        S.active = false;
        S.finished = true;
    }
}

// ---------------------------------------------------------------- one concurrent decode round
static void run_round(std::vector<int>& act) {
    // 1. Enqueue stage 0 for every session on its own stream.
    for (int i : act) {
        Session& S = sess[i];
        session_bind(stages[0], S, 0);
        *S.h_pos = S.pos;
        stage_one_emb(S.h_emb, S.cur_tok);
        ple_window(S, &S.cur_tok, 1, true, S.h_ple);
        S.base_seq[0] = stages[0].cold_seq;
        stages[0].record_forward(S.h_pos, S.h_emb, S.h_ple, 1, !pp2, false,
                                 host_pack, &cpu_pool, S.want_top16);
        CHECK(cudaEventRecord(S.ev[0], S.st[0]));
    }

    // 2. Enqueue stage 1 behind each session's stage-0 event.
    if (pp2) {
        for (int i : act) {
            Session& S = sess[i];
            session_bind(stages[1], S, 1);
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamWaitEvent(S.st[1], S.ev[0], 0));
            CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev, S.R[0].p, stages[0].dev,
                                      (size_t) HCD * sizeof(float), S.st[1]));
            S.base_seq[1] = stages[1].cold_seq;
            stages[1].record_forward(S.h_pos, S.h_emb, S.h_ple, 1, true, false,
                                     host_pack, &cpu_pool, S.want_top16);
            if (!S.fast_greedy && !S.want_top16) {
                CHECK(cudaMemcpyAsync(S.h_logits, stages[1].hlogits.p, (size_t) NVOCAB * sizeof(float),
                                      cudaMemcpyDeviceToHost, S.st[1]));
            }
            CHECK(cudaEventRecord(S.ev[1], S.st[1]));
        }
    } else {
        for (int i : act) {
            Session& S = sess[i];
            if (!S.fast_greedy && !S.want_top16) {
                CHECK(cudaMemcpyAsync(S.h_logits, S.hlogits[0].p, (size_t) NVOCAB * sizeof(float),
                                      cudaMemcpyDeviceToHost, S.st[0]));
            }
            CHECK(cudaEventRecord(S.ev[0], S.st[0]));
        }
    }

    // 3. Feed the shared AVX2 pool. Serving sessions in order is deliberate: the pool is the
    //    bottleneck, so total CPU work is the same either way; what matters is that GPU work for all
    //    sessions was already enqueued, so a slow session does not stall the others' kernels.
    for (int si = 0; si < nstages; ++si) {
        for (int i : act) {
            Session& S = sess[i];
            session_bind(stages[si], S, si);
            stages[si].service_cold_layers(host_pack, &cpu_pool, S.base_seq[si], 1, false);
        }
    }

    // 4. Wait, sample, emit.
    for (int i : act) {
        Session& S = sess[i];
        CHECK(cudaSetDevice(stages.back().dev));
        CHECK(cudaEventSynchronize(S.ev[pp2 ? 1 : 0]));

        S.tokens.push_back(S.cur_tok);
        ++S.pos;

        int tok = sample_next(S);
        if (is_stop_tok(S, tok)) {
            emit_done(S, "stop", 0.0);
            S.active = false;
            S.finished = true;
            S.last_used = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
            continue;
        }
        S.cur_tok = tok;
        emit_tok(S, tok);
        if (S.generated >= S.max_new) {
            emit_done(S, "length", 0.0);
            S.active = false;
            S.finished = true;
            S.last_used = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
        } else if (S.pos >= ctx_max - 8) {
            emit_done(S, "length", 0.0);
            S.active = false;
            S.finished = true;
            S.last_used = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
    }

    // 5. Promote once, after every session's kernels are done: a promotion rewrites a VRAM expert
    //    slot, which must never happen while another session is reading it.
    for (int si = 0; si < nstages; ++si) {
        session_bind(stages[si], sess[act[0]], si);
        stages[si].promote_cold(host_pack, 1, true);
        // Promotion rewrites shared state (d_lut, expert blobs) on one session's stream. Every other
        // session's stream must observe the new LUT before its next record_forward, so drain the device
        // once here. All streams are already idle, so this is a bookkeeping sync, not a stall.
        CHECK(cudaSetDevice(stages[si].dev));
        CHECK(cudaDeviceSynchronize());
    }
}

// ---------------------------------------------------------------- admission
static int pick_slot(const PendingReq& r) {
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < nsess; ++i) {
        Session& S = sess[i];
        if (S.active || S.tokens.empty()) continue;
        if (S.tokens.size() > r.ids.size() - 1) continue;
        bool prefix = true;
        for (size_t k = 0; k < S.tokens.size(); ++k) {
            if (S.tokens[k] != r.ids[k]) { prefix = false; break; }
        }
        if (prefix && S.tokens.size() > best_len) {
            best_len = S.tokens.size();
            best = i;
        }
    }
    if (best >= 0) return best;
    int lru = -1;
    long long stamp = LLONG_MAX;
    for (int i = 0; i < nsess; ++i) {
        if (!sess[i].active && sess[i].last_used < stamp) {
            stamp = sess[i].last_used;
            lru = i;
        }
    }
    return lru;
}

// ---------------------------------------------------------------- boot
int main() {
    const auto t_boot0 = std::chrono::steady_clock::now();
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    pp2 = (ndev >= 2) && (std::getenv("WZY_SINGLE") == nullptr);
    nstages = pp2 ? 2 : 1;
    const int split_layer = pp2 ? (std::getenv("WZY_SPLIT") ? std::atoi(std::getenv("WZY_SPLIT")) : 25) : 48;

    ctx_max = std::getenv("WZY_CTX") ? std::atoi(std::getenv("WZY_CTX")) : KV_VRAM_RING;
    if (ctx_max > KV_VRAM_RING) {
        std::fprintf(stderr,
                     "[wzy-batch] WZY_CTX=%d exceeds the VRAM KV ring (%d). The batch engine has no host "
                     "KV mirror; run long-context traffic on wzy_pp2_engine instead.\n",
                     ctx_max, KV_VRAM_RING);
        return 2;
    }
    const bool explicit_chunk = (std::getenv("WZY_PREFILL_CHUNK") != nullptr);
    explicit_pf_chunk = explicit_chunk;
    prefill_chunk = std::min(PF_MAX, explicit_chunk ? std::atoi(std::getenv("WZY_PREFILL_CHUNK")) : 25600);
    split_pf_chunks = std::getenv("WZY_SPLIT_PF_CHUNKS") ? std::max(1, std::atoi(std::getenv("WZY_SPLIT_PF_CHUNKS"))) : 2;

    int b_env = std::getenv("WZY_BATCH") ? std::atoi(std::getenv("WZY_BATCH")) : 2;
    b_env = std::max(1, std::min(WZY_MAX_SESS, b_env));

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

    stages.resize(nstages);
    if (pp2) {
        init_stage_weights_and_state(stages[0], 0, 0, split_layer, ctx_max, g, h_cos, h_sin);
        init_stage_weights_and_state(stages[1], 1, split_layer, 48, ctx_max, g, h_cos, h_sin);
    } else {
        init_stage_weights_and_state(stages[0], 0, 0, 48, ctx_max, g, h_cos, h_sin);
    }

    // Free buffers that only the speculative / single-stream paths use BEFORE sizing the expert
    // cache: the GDN snapshots alone are ~450 MiB per stage, and `idx_pooled` is ~200 MiB; both are
    // replaced by per-session state here, so handing that VRAM to the expert cache is free hit rate.
    for (int si = 0; si < nstages; ++si) {
        StageGpu& stg = stages[si];
        CHECK(cudaSetDevice(stg.dev));
        if (stg.ssm_snaps.p) { cudaFree(stg.ssm_snaps.p); stg.ssm_snaps.p = nullptr; }
        if (stg.conv_snaps.p) { cudaFree(stg.conv_snaps.p); stg.conv_snaps.p = nullptr; }
        if (stg.ple_snaps.p) { cudaFree(stg.ple_snaps.p); stg.ple_snaps.p = nullptr; }
        if (stg.idx_tail_snaps.p) { cudaFree(stg.idx_tail_snaps.p); stg.idx_tail_snaps.p = nullptr; }
        if (stg.idx_dead_snaps.p) { cudaFree(stg.idx_dead_snaps.p); stg.idx_dead_snaps.p = nullptr; }
        if (stg.idx_pool_snaps.p) { cudaFree(stg.idx_pool_snaps.p); stg.idx_pool_snaps.p = nullptr; }
        if (stg.idx_pooled.p) { cudaFree(stg.idx_pooled.p); stg.idx_pooled.p = nullptr; }
        cudaGetLastError();
    }

    // Reserve VRAM for B resident sessions before the expert cache takes everything.
    {
        size_t per = 0;
        for (const auto& stg : stages) per = std::max(per, session_vram_bytes(stg, ctx_max));
        const size_t leave_mib = (size_t) b_env * ((per + (1u << 20) - 1) >> 20);
        char buf[32];
        std::snprintf(buf, sizeof buf, "%zu", leave_mib);
        setenv("WZY_LEAVE_MIB", buf, 1);
        std::fprintf(stderr, "[wzy-batch] %d session(s) x %.0f MiB reserved from the expert budget\n",
                     b_env, (double) per / 1048576.0);
    }

    for (int si = 0; si < nstages; ++si) alloc_stage_expert_vram(stages[si]);

    // Expert pack: mmap, pre-fault, pin as mapped host memory (read-only source of cold experts).
    const std::string exp_path = std::string(KD) + "/pack/experts.bin";
    const int efd = ::open(exp_path.c_str(), O_RDONLY);
    if (efd < 0) {
        std::fprintf(stderr, "cannot open %s\n", exp_path.c_str());
        return 2;
    }
    struct stat est{};
    fstat(efd, &est);
    const size_t exp_sz = (size_t) est.st_size;
    uint8_t* hp = (uint8_t*) mmap(nullptr, exp_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE, efd, 0);
    ::close(efd);
    if (hp == MAP_FAILED) {
        std::fprintf(stderr, "mmap experts.bin failed\n");
        return 2;
    }
    host_pack = hp;

    if (std::getenv("WZY_NOPIN") == nullptr) {
        std::fprintf(stderr, "[wzy-batch] Pre-faulting & pinning %.2f GiB expert pack across 16 threads...\n",
                     exp_sz / 1073741824.0);
        const int nth = 16;
        std::vector<std::thread> workers;
        for (int tid = 0; tid < nth; ++tid) {
            workers.emplace_back([&, tid]() {
                const size_t pages = exp_sz / 4096;
                volatile uint8_t sink = 0;
                for (size_t pg = tid; pg < pages; pg += nth) sink ^= hp[pg * 4096];
                (void) sink;
            });
        }
        for (auto& th : workers) th.join();
    }
    {
        const size_t CHUNK = (size_t) 256 * PER;
        if (cudaHostRegister(hp, exp_sz, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            cudaGetLastError();
            for (size_t off = 0; off < exp_sz; off += CHUNK) {
                const size_t len = std::min(CHUNK, exp_sz - off);
                cudaHostRegister(hp + off, len, cudaHostRegisterPortable | cudaHostRegisterMapped);
                cudaGetLastError();
            }
        }
    }
    for (int si = 0; si < nstages; ++si) {
        CHECK(cudaSetDevice(stages[si].dev));
        void* dp = nullptr;
        CHECK(cudaHostGetDevicePointer(&dp, hp, 0));
        stages[si].d_pack = (uint8_t*) dp;
    }

    // Prefill VRAM expert slots from the CJK profile (fallback: base profile).
    std::string prof_path = std::getenv("WZY_PROFILE") ? std::getenv("WZY_PROFILE") : "data/expert-profile-cjk.bin";
    if (!std::ifstream(prof_path)) prof_path = "expert-profile-cjk.bin";
    if (!std::ifstream(prof_path)) prof_path = "/home/liaoyang/wzy-flash-serve/expert-profile-cjk.bin";
    const bool using_cjk = (bool) std::ifstream(prof_path);
    if (!using_cjk) {
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
            for (int e = 0; e < NEXP; ++e) ranked.emplace_back(l, e);
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
                CHECK(cudaMemcpyAsync(stg.xblobs + (size_t) vslot * PER, hp + (size_t) gkey * PER,
                                      PER, cudaMemcpyHostToDevice, stg.stream));
            }
            if (using_cjk && stg.h_adapt_ref != nullptr && stg.adapt_cap > 0) {
                std::memset(stg.h_adapt_ref, 1, (size_t) stg.adapt_cap);
            }
            CHECK(cudaMemcpyAsync(stg.d_lut, stg.h_lut.data(), 48 * NEXP * 4, cudaMemcpyHostToDevice, stg.stream));
            CHECK(cudaStreamSynchronize(stg.stream));
            std::fprintf(stderr, "[wzy-batch] GPU%d prefilled %d VRAM expert slots (profile=%s)\n",
                         stg.dev, filled, prof_path.c_str());
        }
    }

    pf_ctx.resize(nstages);
    for (int si = 0; si < nstages; ++si) pf_ctx[si].init(stages[si]);

    // Token embedding (BF16 mmap preferred, GGUF Q8 fallback).
    {
        const TDesc* te = g.find("token_embd.weight");
        g_q_emb = g.at(te);
        g_emb_nb = NE / 32;
        const int bfd = ::open((std::string(KD) + "/token-embd-bf16.gguf").c_str(), O_RDONLY);
        if (bfd >= 0) {
            struct stat bst{};
            if (fstat(bfd, &bst) == 0 && (size_t) bst.st_size >= 256 + (size_t) NVOCAB * NE * 2) {
                uint8_t* em = (uint8_t*) mmap(nullptr, (size_t) bst.st_size, PROT_READ, MAP_PRIVATE, bfd, 0);
                if (em != MAP_FAILED && std::memcmp(em, "GGUF", 4) == 0) {
                    madvise(em, (size_t) bst.st_size, MADV_WILLNEED);
                    volatile uint8_t sink = 0;
                    for (size_t off = 0; off < (size_t) bst.st_size; off += 4096) sink ^= em[off];
                    (void) sink;
                    g_bf_emb = (const uint16_t*) (em + 256);
                }
            }
            ::close(bfd);
        }
    }

    if (!plesrc.open((std::string(KD) + "/ple-fp8.gguf").c_str())) {
        std::fprintf(stderr, "cannot map ple-fp8.gguf\n");
        return 2;
    }

    const int cpu_nth = std::getenv("WZY_CPU_THREADS") ? std::max(1, std::atoi(std::getenv("WZY_CPU_THREADS"))) : 12;
    cpu_pool.init(cpu_nth);

    // Free buffers that only the speculative / single-session paths use, to make room for sessions.
    // (Already released right after weight init; this is the idempotent guard.)

    // Clamp the session count to what actually fits on both GPUs.
    {
        size_t free0 = 0, total0 = 0, free1 = 0, total1 = 0;
        CHECK(cudaSetDevice(stages[0].dev));
        CHECK(cudaMemGetInfo(&free0, &total0));
        if (nstages > 1) {
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaMemGetInfo(&free1, &total1));
        } else {
            free1 = free0;
        }
        const size_t per0 = session_vram_bytes(stages[0], ctx_max);
        const size_t per1 = session_vram_bytes(stages[nstages - 1], ctx_max);
        int fit = (int) std::min((size_t) WZY_MAX_SESS, std::min(free0 / per0, free1 / per1));
        if (fit < b_env) {
            std::fprintf(stderr, "[wzy-batch] WZY_BATCH=%d clamped to %d by free VRAM (%.0f MiB / %.0f MiB)\n",
                         b_env, fit, free0 / 1048576.0, free1 / 1048576.0);
            b_env = fit;
        }
        if (b_env < 1) {
            std::fprintf(stderr, "[wzy-batch] not enough VRAM for a single session; lower WZY_CTX\n");
            return 2;
        }
        nsess = b_env;
    }

    for (int i = 0; i < nsess; ++i) {
        for (int si = 0; si < nstages; ++si) session_alloc(stages[si], sess[i], si, ctx_max);
        sess[i].last_used = 0;
        std::fprintf(stderr, "[wzy-batch] session slot %d allocated\n", i);
    }

    if (pp2) {
        CHECK(cudaSetDevice(stages[0].dev));
        CHECK(cudaEventCreateWithFlags(&ev_s0_done, cudaEventDisableTiming));
        CHECK(cudaSetDevice(stages[1].dev));
        CHECK(cudaEventCreateWithFlags(&ev_s1_rcopied, cudaEventDisableTiming));
    }

    std::signal(SIGUSR1, on_sigusr1_cancel);

    // Warm-up: one dummy row through both stages primes the expert cache and the CUDA context.
    {
        Session& S = sess[0];
        session_bind(stages[0], S, 0);
        int d = 760;
        *S.h_pos = 0;
        stage_one_emb(S.h_emb, d);
        ple_window(S, &d, 1, false, S.h_ple);
        S.base_seq[0] = stages[0].cold_seq;
        stages[0].record_forward(S.h_pos, S.h_emb, S.h_ple, 1, false, false, host_pack, &cpu_pool);
        CHECK(cudaEventRecord(S.ev[0], S.st[0]));
        if (pp2) {
            session_bind(stages[1], S, 1);
            CHECK(cudaSetDevice(stages[1].dev));
            CHECK(cudaStreamWaitEvent(S.st[1], S.ev[0], 0));
            CHECK(cudaMemcpyPeerAsync(stages[1].R.p, stages[1].dev, S.R[0].p, stages[0].dev,
                                      (size_t) HCD * sizeof(float), S.st[1]));
            S.base_seq[1] = stages[1].cold_seq;
            stages[1].record_forward(S.h_pos, S.h_emb, S.h_ple, 1, false, false, host_pack, &cpu_pool);
            CHECK(cudaEventRecord(S.ev[1], S.st[1]));
        }
        for (int si = 0; si < nstages; ++si) {
            session_bind(stages[si], S, si);
            stages[si].service_cold_layers(host_pack, &cpu_pool, S.base_seq[si], 1, false);
        }
        CHECK(cudaSetDevice(stages.back().dev));
        CHECK(cudaEventSynchronize(S.ev[pp2 ? 1 : 0]));
        session_reset(S);
        for (auto& stg : stages) {
            stg.total_lookups = 0;
            stg.total_vram_hits = 0;
            stg.total_promotions = 0;
        }
    }

    int total_vram_slots = 0;
    for (const auto& stg : stages) total_vram_slots += stg.cap;
    const double boot_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_boot0).count();
    say("READY {\"engine\":\"wzy-batch-v1\",\"gpus\":%d,\"split\":%d,\"vram_slots\":%d,"
        "\"mtp\":false,\"ctx_max\":%d,\"boot_s\":%.2f,\"sessions\":%d}\n",
        nstages, split_layer, total_vram_slots, ctx_max, boot_s, nsess);
    // ---------------------------------------------------------------- scheduler
    std::deque<PendingReq> pending;
    std::deque<int> cancels;
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> quit{false};

    std::thread reader([&]() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line == "PING") {
                say("PONG\n");
                continue;
            }
            if (line == "QUIT" || line == "EXIT") {
                quit.store(true);
                cv.notify_all();
                break;
            }
            if (line.rfind("CANCEL ", 0) == 0) {
                const int rid = std::atoi(line.c_str() + 7);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    cancels.push_back(rid);
                }
                cv.notify_all();
                continue;
            }
            if (line[0] != '{') continue;

            PendingReq r;
            r.req = (int) parse_json_num(line, "\"req\"", 0);
            parse_json_ints(line, "\"ids\"", r.ids);
            if (r.ids.empty()) continue;
            std::vector<int> stops;
            parse_json_ints(line, "\"stop\"", stops);
            if (!stops.empty()) r.cfg.stops = stops;
            r.cfg.max_new = (int) parse_json_num(line, "\"max_tokens\"", 512);
            r.cfg.temp = (float) parse_json_num(line, "\"temperature\"", 0.3);
            r.cfg.top_p = (float) parse_json_num(line, "\"top_p\"", 0.75);
            r.cfg.top_k = (int) parse_json_num(line, "\"top_k\"", 15);
            r.cfg.min_p = (float) parse_json_num(line, "\"min_p\"", 0.02);
            r.cfg.rep_pen = (float) parse_json_num(line, "\"repetition_penalty\"", 1.0);
            parse_json_str(line, "\"emb_file\"", r.emb_file);
            if (!r.emb_file.empty()) {
                parse_json_ints(line, "\"emb_starts\"", r.emb_starts);
                parse_json_ints(line, "\"emb_counts\"", r.emb_counts);
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                pending.push_back(std::move(r));
            }
            cv.notify_all();
        }
        quit.store(true);
        cv.notify_all();
    });

    while (!quit.load()) {
        // Cancellation requests: mark the matching session or drop a queued request.
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int rid : cancels) {
                for (int i = 0; i < nsess; ++i) {
                    if (sess[i].active && sess[i].id == rid) sess[i].cancelled = true;
                }
                for (auto it = pending.begin(); it != pending.end();) {
                    if (it->req == rid) it = pending.erase(it);
                    else ++it;
                }
            }
            cancels.clear();
        }
        if (g_cancel_req) {
            g_cancel_req = 0;
            for (int i = 0; i < nsess; ++i) if (sess[i].active) sess[i].cancelled = true;
        }

        // Admit as many queued requests as slots allow. Prefill is blocking: it uses the shared
        // prefill scratch and borrows VRAM expert slots, so it must run with all streams idle.
        while (true) {
            PendingReq r;
            int slot = -1;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (pending.empty()) break;
                slot = pick_slot(pending.front());
                if (slot < 0) break;
                r = pending.front();
                pending.pop_front();
            }
            prefill_session(sess[slot], r);
            sess[slot].last_used = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        std::vector<int> act;
        for (int i = 0; i < nsess; ++i) {
            Session& S = sess[i];
            if (!S.active) continue;
            if (S.cancelled) {
                emit_done(S, "cancelled", 0.0);
                S.active = false;
                S.finished = true;
                S.last_used = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count();
                continue;
            }
            act.push_back(i);
        }
        if (act.empty()) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(50));
            continue;
        }
        run_round(act);
    }

    reader.join();
    return 0;
}
