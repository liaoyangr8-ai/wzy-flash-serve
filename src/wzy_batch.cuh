// wzy_batch.cuh -- per-session state, scratch and pointer binding for the concurrent multi-session engine.
//
// The v2 layer executor (StageGpu::record_forward) is reused VERBATIM. Every buffer it touches is a
// member pointer, so a session is just a different set of pointers: session_bind() swaps them in, and
// record_forward then runs on that session's stream, KV ring, GDN state, cold-expert mailbox and
// logits. No kernel is modified, so a session's arithmetic is bit-identical to the single-stream path.
#pragma once

#include "wzy_setup.cuh"

static constexpr int WZY_MAX_SESS = 4;

struct Session {
    int id = -1;
    bool active = false;       // has an admitted request and is decoding
    bool cancelled = false;
    bool finished = false;

    std::vector<int> tokens;   // prompt + every token already fed to the model
    int pos = 0;               // position of `cur_tok` (the next token to feed)
    int cur_tok = -1;          // sampled, not yet fed
    int generated = 0;
    int max_new = 512;

    GenCfg cfg;
    bool fast_greedy = false;
    bool want_top16 = false;
    int ple_prev[2] = {-1, -1};
    int prompt_len = 0;        // tokens.size() right after prefill: start of generated text
    int reused_tokens = 0;     // prompt prefix served from the resident session state
    long long last_used = 0;   // LRU stamp for slot reuse
    double t_first = 0.0;      // wall clock of the first generated token (decode_ms baseline)

    // Optional vision rows for this request (SVE1 file), one run per image in prompt order.
    std::vector<float> img_emb;
    std::vector<int> img_start, img_count;

    uint32_t base_seq[2] = {0, 0};   // cold-expert sequence base captured before each record_forward

    cudaStream_t st[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};

    // Host staging (pinned / mapped). Per session: async H2D of h_emb must not be clobbered by the
    // next session's staging, so these cannot be shared. h_emb/h_ple hold up to MAX_W rows because the
    // short-prompt prefill path feeds up to MAX_W tokens per step.
    float* h_emb = nullptr;      // [MAX_W * NE]
    float* h_ple = nullptr;      // [MAX_W * NE]
    int* h_pos = nullptr;        // [1]
    float* h_logits = nullptr;   // [NVOCAB]
    int* h_greedy = nullptr;     // [MAX_W]
    float* h_top16v = nullptr;   // [MAX_W * 16]
    int* h_top16i = nullptr;     // [MAX_W * 16]
    int* d_greedy = nullptr;     // mapped aliases
    float* d_top16v = nullptr;
    int* d_top16i = nullptr;

    ColdLayerReq* h_cold_req = nullptr;   // [48]
    ColdLayerReq* d_cold_req = nullptr;
    float* h_cold_out[2][48] = {};
    float* d_cold_out[2][48] = {};

    // ---- device state, per stage ----
    Dev conv[2][48];
    Dev ssm[2][48];
    Dev ple_hist[2];
    Dev hlogits[2];
    Dev idx_tail[2], idx_dead[2], idx_pooled[2], idx_scores[2];
    int* sel_ids[2] = {nullptr, nullptr};
    int* sel_cnt[2] = {nullptr, nullptr};
    signed char* vk_q[2] = {nullptr, nullptr};
    signed char* vv_q[2] = {nullptr, nullptr};
    unsigned short* vk_s[2] = {nullptr, nullptr};
    unsigned short* vv_s[2] = {nullptr, nullptr};
    int* dpos[2] = {nullptr, nullptr};

    // ---- per-stage scratch (one row) ----
    Dev R[2], R_pre_head[2], xn[2], lo[2], mixed[2], mixed2[2], inject[2], block[2];
    Dev qkv[2], convout[2], z[2], decay[2], bt[2], on[2];
    Dev qg[2], kk[2], vv[2], attn[2], idx_raw[2], q_idx[2];
    Dev slog[2], sids[2], swts[2], res_swts[2], sff[2], sup[2], sshared[2], sgate[2], tmp_ff[2];
    Dev parts[2], comb[2], ffact[2], exsc[2];
    Dev ple_q[2], ple_k[2], ple_v[2], ple_gat[2], ple_n[2], d_ple_emb[2];

    signed char* xq8[2] = {nullptr, nullptr};
    float* xs8[2] = {nullptr, nullptr};
    signed char* eq32[2] = {nullptr, nullptr};
    signed char* eq16[2] = {nullptr, nullptr};
    float* es32[2] = {nullptr, nullptr};
    float* es16[2] = {nullptr, nullptr};
    signed char* dq16[2] = {nullptr, nullptr};
    float* ds16[2] = {nullptr, nullptr};
    int* grp_slot[2] = {nullptr, nullptr};
    int* grp_cnt[2] = {nullptr, nullptr};
    int* grp_e[2] = {nullptr, nullptr};
    int* d_ngroups[2] = {nullptr, nullptr};
    int* dslots[2] = {nullptr, nullptr};
    int* mkey[2] = {nullptr, nullptr};
    int* dmisses[2] = {nullptr, nullptr};
};

static inline Dev wzy_dev(size_t n) {
    Dev d;
    d.n = n;
    CHECK(cudaMalloc(&d.p, n * sizeof(float)));
    return d;
}

static inline Dev wzy_dev0(size_t n) {
    Dev d = wzy_dev(n);
    CHECK(cudaMemset(d.p, 0, n * sizeof(float)));
    return d;
}

static inline void* wzy_bytes(size_t n) {
    void* p = nullptr;
    CHECK(cudaMalloc(&p, n));
    CHECK(cudaMemset(p, 0, n));
    return p;
}

// VRAM a single session needs on one stage, in bytes. Used to clamp the session count to what fits
// after the expert cache has been sized (WZY_LEAVE_MIB must hold this much back per session).
static size_t session_vram_bytes(const StageGpu& stg, int ctx_max) {
    const size_t RW = MAX_W;
    // The Tensor-Core prefill path scores `(p_start + B) / IDX_BLK + 4` blocks, which is 2 more than
    // the decode path's `ctx_max / IDX_BLK + 2` clamp; size the pool for the larger of the two.
    const int max_b = ctx_max / IDX_BLK + 8;
    size_t b = 0;

    // Persistent state
    b += (size_t) stg.n_qsa * KV_VRAM_RING * KVD * 2;                                  // INT8 K+V ring
    b += (size_t) stg.n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * 2 * sizeof(unsigned short);
    b += (size_t) stg.n_gdn * QKV * 3 * sizeof(float);                                 // conv state
    b += (size_t) stg.n_gdn * VH * ST * ST * sizeof(float);                            // ssm state
    b += (size_t) stg.n_qsa * ((IDX_BLK - 1) * IDX_DIM + IDX_DIM + (size_t) max_b * IDX_DIM) * sizeof(float);
    b += (size_t) max_b * RW * sizeof(float);                                          // idx_scores
    b += (size_t) IDX_TOPK * RW * sizeof(int) + RW * sizeof(int);                      // sel_ids / sel_cnt
    b += (size_t) NVOCAB * sizeof(float);                                              // logits row
    b += (size_t) HCD * PLE_HIST * sizeof(float);                                      // PLE history

    // Row-major scratch, RW rows (decode uses 1, short prefill uses up to MAX_W)
    const size_t row_floats =
        (size_t) HCD * 3 + HCLR + (size_t) NE * 5 + HC + (size_t) QKV * 2 + (size_t) GV * 2 +
        VH * 2 + (size_t) QD * 3 + (size_t) KVD * 2 + IDX_DIM + (size_t) IDX_HEADS * IDX_DIM +
        NEXP + (size_t) TOPK * 3 + (size_t) TOPK * 3 + (size_t) NFF * 3 + 1 +
        (size_t) TOPK * NE + (size_t) TOPK * NFF + (size_t) TOPK * 3 +
        (size_t) HCD * 4 + (size_t) NE * 2;
    b += RW * row_floats * sizeof(float);
    b += RW * ((size_t) HCD + (size_t) NE * 2 + (size_t) TOPK * NFF);                  // xq8/eq32/eq16/dq16 bytes
    b += RW * ((size_t) (HCD / 32) + (size_t) (NE / 32) + (size_t) (NE / 16) +
               (size_t) (TOPK * NFF / 16)) * sizeof(float);                            // scale vectors
    b += (size_t) TOPK * GRP_MAX * RW * sizeof(int);                                   // grp_e
    b += (size_t) 2048 * sizeof(int);                                                  // mkey
    b += 4u << 20;                                                                     // slack
    return b;
}

static void session_alloc(StageGpu& stg, Session& S, int si, int ctx_max) {
    CHECK(cudaSetDevice(stg.dev));
    CHECK(cudaStreamCreateWithFlags(&S.st[si], cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&S.ev[si], cudaEventDisableTiming));

    const int n_qsa = stg.n_qsa;
    // Must cover the prefill scorer's `(p_start + B) / IDX_BLK + 4` block range (see session_vram_bytes).
    const int max_b = ctx_max / IDX_BLK + 8;

    // Persistent state
    for (int l = stg.l_begin; l < stg.l_end; ++l) {
        if (!stg.L[l].qsa) {
            S.conv[si][l] = wzy_dev0((size_t) QKV * 3);
            S.ssm[si][l] = wzy_dev0((size_t) VH * ST * ST);
        }
    }
    if (stg.has_ple) S.ple_hist[si] = wzy_dev0((size_t) HCD * PLE_HIST);

    S.vk_q[si] = (signed char*) wzy_bytes((size_t) n_qsa * KV_VRAM_RING * KVD);
    S.vv_q[si] = (signed char*) wzy_bytes((size_t) n_qsa * KV_VRAM_RING * KVD);
    S.vk_s[si] = (unsigned short*) wzy_bytes((size_t) n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * sizeof(unsigned short));
    S.vv_s[si] = (unsigned short*) wzy_bytes((size_t) n_qsa * KV_VRAM_RING * KV_SC_PER_CELL * sizeof(unsigned short));

    S.idx_tail[si] = wzy_dev0((size_t) n_qsa * (IDX_BLK - 1) * IDX_DIM);
    S.idx_dead[si] = wzy_dev0((size_t) n_qsa * IDX_DIM);
    S.idx_pooled[si] = wzy_dev0((size_t) n_qsa * max_b * IDX_DIM);
    S.idx_scores[si] = wzy_dev0((size_t) max_b * MAX_W);
    S.sel_ids[si] = (int*) wzy_bytes((size_t) IDX_TOPK * MAX_W * sizeof(int));
    S.sel_cnt[si] = (int*) wzy_bytes((size_t) MAX_W * sizeof(int));

    CHECK(cudaMalloc(&S.dpos[si], sizeof(int)));
    CHECK(cudaMemset(S.dpos[si], 0, sizeof(int)));

    // Scratch: up to MAX_W rows. Decode feeds one row, but the short-prompt prefill path pushes up to
    // MAX_W tokens through the same kernels, so every row-major buffer must be sized for MAX_W.
    const size_t RW = MAX_W;
    S.R[si] = wzy_dev0((size_t) HCD * RW);
    S.R_pre_head[si] = wzy_dev0((size_t) HCD * RW);
    S.xn[si] = wzy_dev0((size_t) HCD * RW);
    S.lo[si] = wzy_dev0((size_t) HCLR * RW);
    S.mixed[si] = wzy_dev0((size_t) NE * RW);
    S.mixed2[si] = wzy_dev0((size_t) NE * RW);
    S.block[si] = wzy_dev0((size_t) NE * RW);
    S.sshared[si] = wzy_dev0((size_t) NE * RW);
    S.comb[si] = wzy_dev0((size_t) NE * RW);
    S.inject[si] = wzy_dev0((size_t) HC * RW);
    S.qkv[si] = wzy_dev0((size_t) QKV * RW);
    S.convout[si] = wzy_dev0((size_t) QKV * RW);
    S.z[si] = wzy_dev0((size_t) GV * RW);
    S.on[si] = wzy_dev0((size_t) GV * RW);
    S.decay[si] = wzy_dev0((size_t) VH * RW);
    S.bt[si] = wzy_dev0((size_t) VH * RW);
    S.qg[si] = wzy_dev0((size_t) 2 * QD * RW);
    S.kk[si] = wzy_dev0((size_t) KVD * RW);
    S.vv[si] = wzy_dev0((size_t) KVD * RW);
    S.attn[si] = wzy_dev0((size_t) QD * RW);
    S.idx_raw[si] = wzy_dev0((size_t) IDX_DIM * RW);
    S.q_idx[si] = wzy_dev0((size_t) IDX_HEADS * IDX_DIM * RW);
    S.slog[si] = wzy_dev0((size_t) NEXP * RW);
    S.sids[si] = wzy_dev0((size_t) TOPK * RW);
    S.swts[si] = wzy_dev0((size_t) TOPK * RW);
    S.res_swts[si] = wzy_dev0((size_t) TOPK * RW);
    S.sff[si] = wzy_dev0((size_t) NFF * RW);
    S.sup[si] = wzy_dev0((size_t) NFF * RW);
    S.tmp_ff[si] = wzy_dev0((size_t) NFF * RW);
    S.sgate[si] = wzy_dev0(RW);
    S.parts[si] = wzy_dev0((size_t) TOPK * NE * RW);
    S.ffact[si] = wzy_dev0((size_t) TOPK * NFF * RW);
    S.exsc[si] = wzy_dev0((size_t) TOPK * 3 * RW);
    S.hlogits[si] = wzy_dev0((size_t) NVOCAB);

    if (stg.has_ple) {
        S.ple_q[si] = wzy_dev0((size_t) HCD * RW);
        S.ple_k[si] = wzy_dev0((size_t) HCD * RW);
        S.ple_v[si] = wzy_dev0((size_t) NE * RW);
        S.ple_gat[si] = wzy_dev0((size_t) HCD * RW);
        S.ple_n[si] = wzy_dev0((size_t) HCD * RW);
        S.d_ple_emb[si] = wzy_dev0((size_t) NE * RW);
    }

    S.xq8[si] = (signed char*) wzy_bytes((size_t) HCD * RW);
    S.xs8[si] = (float*) wzy_bytes((size_t) (HCD / 32) * RW * sizeof(float));
    S.eq32[si] = (signed char*) wzy_bytes((size_t) NE * RW);
    S.eq16[si] = (signed char*) wzy_bytes((size_t) NE * RW);
    S.es32[si] = (float*) wzy_bytes((size_t) (NE / 32) * RW * sizeof(float));
    S.es16[si] = (float*) wzy_bytes((size_t) (NE / 16) * RW * sizeof(float));
    S.dq16[si] = (signed char*) wzy_bytes((size_t) TOPK * NFF * RW);
    S.ds16[si] = (float*) wzy_bytes((size_t) (TOPK * NFF / 16) * RW * sizeof(float));
    S.grp_slot[si] = (int*) wzy_bytes((size_t) TOPK * RW * sizeof(int));
    S.grp_cnt[si] = (int*) wzy_bytes((size_t) TOPK * RW * sizeof(int));
    S.grp_e[si] = (int*) wzy_bytes((size_t) TOPK * GRP_MAX * RW * sizeof(int));
    S.d_ngroups[si] = (int*) wzy_bytes(sizeof(int));
    S.dslots[si] = (int*) wzy_bytes((size_t) TOPK * RW * sizeof(int));
    S.mkey[si] = (int*) wzy_bytes((size_t) 2048 * sizeof(int));
    S.dmisses[si] = (int*) wzy_bytes(sizeof(int));

    // Host staging
    CHECK(cudaHostAlloc((void**) &S.h_emb, (size_t) MAX_W * NE * sizeof(float), cudaHostAllocPortable));
    CHECK(cudaHostAlloc((void**) &S.h_ple, (size_t) MAX_W * NE * sizeof(float), cudaHostAllocPortable));
    CHECK(cudaHostAlloc((void**) &S.h_pos, sizeof(int), cudaHostAllocPortable));
    CHECK(cudaHostAlloc((void**) &S.h_logits, (size_t) NVOCAB * sizeof(float), cudaHostAllocPortable));
    std::memset(S.h_emb, 0, (size_t) MAX_W * NE * sizeof(float));
    std::memset(S.h_ple, 0, (size_t) MAX_W * NE * sizeof(float));
    std::memset(S.h_logits, 0, (size_t) NVOCAB * sizeof(float));
    *S.h_pos = 0;

    CHECK(cudaHostAlloc((void**) &S.h_greedy, (size_t) MAX_W * sizeof(int),
                        cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK(cudaHostAlloc((void**) &S.h_top16v, (size_t) MAX_W * 16 * sizeof(float),
                        cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK(cudaHostAlloc((void**) &S.h_top16i, (size_t) MAX_W * 16 * sizeof(int),
                        cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK(cudaHostGetDevicePointer((void**) &S.d_greedy, S.h_greedy, 0));
    CHECK(cudaHostGetDevicePointer((void**) &S.d_top16v, S.h_top16v, 0));
    CHECK(cudaHostGetDevicePointer((void**) &S.d_top16i, S.h_top16i, 0));

    CHECK(cudaHostAlloc((void**) &S.h_cold_req, 48 * sizeof(ColdLayerReq),
                        cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset((void*) S.h_cold_req, 0, 48 * sizeof(ColdLayerReq));
    CHECK(cudaHostGetDevicePointer((void**) &S.d_cold_req, (void*) S.h_cold_req, 0));

    for (int l = stg.l_begin; l < stg.l_end; ++l) {
        CHECK(cudaHostAlloc((void**) &S.h_cold_out[si][l], (size_t) MAX_W * NE * sizeof(float),
                            cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(S.h_cold_out[si][l], 0, (size_t) MAX_W * NE * sizeof(float));
        CHECK(cudaHostGetDevicePointer((void**) &S.d_cold_out[si][l], S.h_cold_out[si][l], 0));
    }
}

// Swap a session's pointers into a stage. record_forward() and service_one_cold_layer() are then
// operating purely on this session. Host-side pointer writes; safe to call between async launches
// because every kernel argument is captured at launch time.
static void session_bind(StageGpu& stg, Session& S, int si) {
    CHECK(cudaSetDevice(stg.dev));
    stg.stream = S.st[si];
    cublasSetStream(stg.cublas, stg.stream);
    stg.dpos = S.dpos[si];

    for (int l = 0; l < 48; ++l) {
        stg.conv[l] = S.conv[si][l];
        stg.ssm[l] = S.ssm[si][l];
    }
    // Snapshots belong to the speculative verify path; the batch engine decodes one row per step and
    // never commits a window, so leaving them null turns any accidental use into a loud failure.
    stg.conv_snaps.p = nullptr;
    stg.ssm_snaps.p = nullptr;
    stg.ple_snaps.p = nullptr;
    stg.idx_tail_snaps.p = nullptr;
    stg.idx_dead_snaps.p = nullptr;
    stg.idx_pool_snaps.p = nullptr;

    stg.vk_q = S.vk_q[si];
    stg.vv_q = S.vv_q[si];
    stg.vk_s = S.vk_s[si];
    stg.vv_s = S.vv_s[si];
    stg.hk_q = nullptr;   // ctx_max <= KV_VRAM_RING is enforced by the engine: no host mirror needed
    stg.hv_q = nullptr;
    stg.hk_s = nullptr;
    stg.hv_s = nullptr;

    stg.idx_tail = S.idx_tail[si];
    stg.idx_dead = S.idx_dead[si];
    stg.idx_pooled = S.idx_pooled[si];
    stg.idx_pool_stride = (int) (stg.idx_pooled.n / (size_t) std::max(1, stg.n_qsa) / IDX_DIM);
    stg.idx_scores = S.idx_scores[si];
    stg.sel_ids = S.sel_ids[si];
    stg.sel_cnt = S.sel_cnt[si];

    stg.R = S.R[si];
    stg.R_pre_head = S.R_pre_head[si];
    stg.xn = S.xn[si];
    stg.lo = S.lo[si];
    stg.mixed = S.mixed[si];
    stg.mixed2 = S.mixed2[si];
    stg.inject = S.inject[si];
    stg.block = S.block[si];
    stg.qkv = S.qkv[si];
    stg.convout = S.convout[si];
    stg.z = S.z[si];
    stg.decay = S.decay[si];
    stg.bt = S.bt[si];
    stg.on = S.on[si];
    stg.qg = S.qg[si];
    stg.kk = S.kk[si];
    stg.vv = S.vv[si];
    stg.attn = S.attn[si];
    stg.idx_raw = S.idx_raw[si];
    stg.q_idx = S.q_idx[si];
    stg.slog = S.slog[si];
    stg.sids = S.sids[si];
    stg.swts = S.swts[si];
    stg.res_swts = S.res_swts[si];
    stg.sff = S.sff[si];
    stg.sup = S.sup[si];
    stg.sshared = S.sshared[si];
    stg.sgate = S.sgate[si];
    stg.tmp_ff = S.tmp_ff[si];
    stg.parts = S.parts[si];
    stg.comb = S.comb[si];
    stg.ffact = S.ffact[si];
    stg.exsc = S.exsc[si];

    stg.xq8 = S.xq8[si];
    stg.xs8 = S.xs8[si];
    stg.eq32 = S.eq32[si];
    stg.es32 = S.es32[si];
    stg.eq16 = S.eq16[si];
    stg.es16 = S.es16[si];
    stg.dq16 = S.dq16[si];
    stg.ds16 = S.ds16[si];

    stg.grp_slot = S.grp_slot[si];
    stg.grp_cnt = S.grp_cnt[si];
    stg.grp_e = S.grp_e[si];
    stg.d_ngroups = S.d_ngroups[si];
    stg.dslots = S.dslots[si];
    stg.mkey = S.mkey[si];
    stg.dmisses = S.dmisses[si];

    if (stg.has_ple) {
        stg.ple_hist = S.ple_hist[si];
        stg.d_ple_emb = S.d_ple_emb[si];
        stg.ple_q = S.ple_q[si];
        stg.ple_k = S.ple_k[si];
        stg.ple_v = S.ple_v[si];
        stg.ple_gat = S.ple_gat[si];
        stg.ple_n = S.ple_n[si];
    }

    stg.hlogits = S.hlogits[si];
    stg.h_greedy_ids = S.h_greedy;
    stg.d_greedy_ids = S.d_greedy;
    stg.h_top16_val = S.h_top16v;
    stg.d_top16_val = S.d_top16v;
    stg.h_top16_id = S.h_top16i;
    stg.d_top16_id = S.d_top16i;

    stg.h_cold_req = S.h_cold_req;
    stg.d_cold_req = S.d_cold_req;
    for (int l = 0; l < 48; ++l) {
        stg.h_cold_out[l] = S.h_cold_out[si][l];
        stg.d_cold_out[l] = S.d_cold_out[si][l];
    }
}
