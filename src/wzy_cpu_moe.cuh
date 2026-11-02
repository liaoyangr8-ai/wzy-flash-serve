// wzy_cpu_moe.cuh -- Persistent 16-Thread AVX2+FMA Host Cold-Expert Pool + GPU/CPU Split-MoE Kernels
// Eliminates 100% of scattered PCIe zero-copy GPU stalls during Decode & Speculative Verify!
#pragma once
#include "wzy_kernels_v2.cuh"
#include <atomic>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>
static constexpr int MAX_COLD_GROUPS = MAX_W * TOPK; // 80

struct alignas(64) ColdLayerReq {
    volatile uint32_t ready_seq;
    volatile uint32_t cpu_done_seq;
    volatile int n_cold;
    int T;
    int layer;
    int cold_gkey[MAX_COLD_GROUPS];
    int cold_cnt[MAX_COLD_GROUPS];
    int cold_tok[MAX_COLD_GROUPS][GRP_MAX];
    float cold_wt[MAX_COLD_GROUPS][GRP_MAX];
    // Pre-permuted xq16: within each 64-byte block, sub-blocks [s0, s1, s2, s3] are stored as [s0, s2, s1, s3]
    // so AVX2 256-bit loads match `_mm256_unpacklo/hi_epi64` with ZERO permutes in the inner loop!
    alignas(64) signed char xq_perm[MAX_W * NE];
    alignas(64) float xs16[MAX_W * (NE / 16)];
};

// Device kernel: splits `total_k = T * TOPK` routed selections into:
//   1) VRAM-resident groups (`slot >= 0`) -> `grp_slot, grp_cnt, grp_e, d_ngroups` (processed by GPU at 616 GB/s GDDR6)
//   2) Cold host groups (`slot < 0`) -> UVA pinned `req` (processed in parallel by 16 AVX2+FMA CPU threads at 19 GB/s DDR3)
// Also zeroes `res_swts[e]` when `dslots[e] < 0` so `moe_combine_shared_multi` on the GPU sums ONLY resident experts!
__global__ void split_route_hit_cold_kernel(
    const int* __restrict__ sids, const float* __restrict__ swts, int T,
    const int* __restrict__ d_lut, int layer,
    int* __restrict__ dslots, float* __restrict__ res_swts,
    int* __restrict__ grp_slot, int* __restrict__ grp_cnt, int* __restrict__ grp_e, int* __restrict__ d_ngroups,
    const signed char* __restrict__ eq16, const float* __restrict__ es16,
    ColdLayerReq* __restrict__ req, uint32_t seq_id,
    int static_cap = 0, uint8_t* __restrict__ d_adapt_ref = nullptr, int* __restrict__ d_ncold_out = nullptr) {
    const int eid = threadIdx.x; // 0..511
    const int total_k = T * TOPK;
    __shared__ int s_ncold;
    if (eid == 0) {
        *d_ngroups = 0;
        s_ncold = 0;
    }
    __syncthreads();

    for (int e = eid; e < total_k; e += 512) {
        const int gkey = layer * NEXP + sids[e];
        const int sl = d_lut[gkey];
        dslots[e] = (sl >= 0) ? sl : -(1 + gkey);
        res_swts[e] = (sl >= 0) ? swts[e] : 0.0f;
        if (d_adapt_ref != nullptr && sl >= static_cap) {
            const int r = sl - static_cap;
            // Fire-and-forget posted write (NO PCIe read of `d_adapt_ref[r] == 0` -> zero GPU stall!)
            if ((unsigned) r < 8192u) d_adapt_ref[r] = 1;
        }
    }
    __syncthreads();

    if (eid < NEXP) {
        const int gkey = layer * NEXP + eid;
        const int sl = d_lut[gkey];
        int buf_e[GRP_MAX];
        int c = 0;
        for (int e = 0; e < total_k; ++e) {
            if (sids[e] == eid) {
                buf_e[c++] = e;
                if (c == GRP_MAX) {
                    if (sl >= 0) {
                        const int g = atomicAdd(d_ngroups, 1);
                        grp_slot[g] = sl;
                        grp_cnt[g] = c;
                        #pragma unroll
                        for (int j = 0; j < GRP_MAX; ++j) grp_e[g * GRP_MAX + j] = buf_e[j];
                    } else {
                        const int cg = atomicAdd(&s_ncold, 1);
                        if (cg < MAX_COLD_GROUPS) {
                            req->cold_gkey[cg] = gkey;
                            req->cold_cnt[cg] = c;
                            #pragma unroll
                            for (int j = 0; j < GRP_MAX; ++j) {
                                req->cold_tok[cg][j] = buf_e[j] / TOPK;
                                req->cold_wt[cg][j] = swts[buf_e[j]];
                            }
                        }
                    }
                    c = 0;
                }
            }
        }
        if (c > 0) {
            if (sl >= 0) {
                const int g = atomicAdd(d_ngroups, 1);
                grp_slot[g] = sl;
                grp_cnt[g] = c;
                for (int j = 0; j < c; ++j) grp_e[g * GRP_MAX + j] = buf_e[j];
            } else {
                const int cg = atomicAdd(&s_ncold, 1);
                if (cg < MAX_COLD_GROUPS) {
                    req->cold_gkey[cg] = gkey;
                    req->cold_cnt[cg] = c;
                    for (int j = 0; j < c; ++j) {
                        req->cold_tok[cg][j] = buf_e[j] / TOPK;
                        req->cold_wt[cg][j] = swts[buf_e[j]];
                    }
                }
            }
        }
    }
    __syncthreads();

    const int ncold = min(s_ncold, MAX_COLD_GROUPS);
    if (ncold > 0) {
        // Copy & pre-permute eq16 [s0, s1, s2, s3] -> [s0, s2, s1, s3] per 64-byte block into UVA pinned `req->xq_perm`
        const int n_blk64 = T * (NE / 64);
        for (int b = eid; b < n_blk64; b += 512) {
            const int4* src = (const int4*) (eq16 + b * 64);
            int4* dst = (int4*) (req->xq_perm + b * 64);
            dst[0] = src[0]; // s0 (0..15)
            dst[1] = src[2]; // s2 (32..47)
            dst[2] = src[1]; // s1 (16..31)
            dst[3] = src[3]; // s3 (48..63)
        }
        const int n_sc4 = T * (NE / 64);
        for (int b = eid; b < n_sc4; b += 512) {
            *((float4*) req->xs16 + b) = *((const float4*) es16 + b);
        }
        // Every GPU thread that wrote mapped host memory (`cold_gkey/cnt/tok/wt`, `xq_perm`, `xs16`)
        // must flush its own writes to system scope BEFORE thread 0 publishes `ready_seq = seq_id`!
        __threadfence_system();
    }
    __syncthreads();

    if (eid == 0) {
        if (d_ncold_out != nullptr) *d_ncold_out = ncold;
        req->n_cold = ncold;
        req->T = T;
        req->layer = layer;
        __threadfence_system();
        req->ready_seq = seq_id;
    }
}

__global__ void add_cold_out_kernel(float* __restrict__ comb, const float* __restrict__ cold_out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) comb[i] += cold_out[i];
}

__global__ void wait_and_add_cold_out_kernel(float* __restrict__ comb, const volatile float* __restrict__ cold_out,
                                             const volatile ColdLayerReq* __restrict__ req, uint32_t seq_id, int n,
                                             const int* __restrict__ d_ncold = nullptr) {
    if (d_ncold != nullptr) {
        if (*d_ncold <= 0) return; // 100% VRAM hit -> 1-cycle GPU L2 cache fast exit (ZERO PCIe traffic!)
    } else {
        int ncold = 0;
        asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ncold) : "l"(&req->n_cold) : "memory");
        if (ncold <= 0) return;
    }
    __shared__ uint32_t s_done;
    if (threadIdx.x == 0) {
        uint32_t done = 0;
        do {
            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(done) : "l"(&req->cpu_done_seq) : "memory");
        } while (done != seq_id);
        __threadfence_system();
        s_done = done;
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && s_done == seq_id) {
        float cv = 0.f;
        asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(cv) : "l"(cold_out + i) : "memory");
        comb[i] += cv;
    }
}

// ==================================================================
// HOST AVX2 + FMA NVFP4 COLD EXPERT ENGINE
// ==================================================================
struct alignas(64) CpuMoePool {
    int n_threads = 12;
    float ue_tab[256];
    std::vector<std::thread> workers;
    std::atomic<int> worker_tids[32];
    int host_tid = -1;
    std::atomic<int> s1_launcher_tid{-1};
    std::atomic<bool> stop_flag{false};
    alignas(64) std::atomic<uint32_t> work_seq{0};
    alignas(64) std::atomic<int> done_count{0};
    alignas(64) std::atomic<int> next_task1{0};
    alignas(64) std::atomic<int> next_task2{0};
    // Current job pointers
    const ColdLayerReq* cur_req = nullptr;
    const uint8_t* cur_pack = nullptr;
    float* cur_out = nullptr;
    std::atomic<bool> sleep_mode{false};
    void set_sleep(bool v) { sleep_mode.store(v, std::memory_order_release); }

    // Intermediate SwiGLU activations quantized to int8 (pre-permuted [s0, s2, s1, s3]!) + float scales:
    // Indexed by `[cg][j]` (`cg < MAX_COLD_GROUPS`, `j < GRP_MAX`)
    alignas(64) signed char dq_perm[MAX_COLD_GROUPS][GRP_MAX][NFF];
    alignas(64) float ds16[MAX_COLD_GROUPS][GRP_MAX][NFF / 16];

    void init(int nth = 12) {
        n_threads = std::max(2, std::min(15, nth));
        for (int i = 0; i < 32; ++i) worker_tids[i].store(-1, std::memory_order_relaxed);
        for (int x = 0; x < 256; ++x) {
            const int u = x & 0x7F;
            if (u == 0) { ue_tab[x] = 0.f; continue; }
            const int e = (u >> 3) & 0xF, m = u & 7;
            const float raw = (e == 0) ? std::ldexp((float) m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
            ue_tab[x] = raw * 0.5f;
        }
        stop_flag.store(false, std::memory_order_relaxed);
        work_seq.store(0, std::memory_order_relaxed);
        done_count.store(0, std::memory_order_relaxed);

        // Pin host controller thread (`tid = 0`) to dedicated physical core 15 so it NEVER collides with workers!
        host_tid = (int) syscall(SYS_gettid);
        {
            cpu_set_t host_set;
            CPU_ZERO(&host_set);
            CPU_SET(15, &host_set);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &host_set);
        }

        // Spawn `n_threads - 1` background threads pinned to physical cores `0 .. n_threads - 2` (leaving cores 11..14 & 27..31 for CUDA/OS!)
        for (int tid = 1; tid < n_threads; ++tid) {
            workers.emplace_back([this, tid]() {
                worker_tids[tid].store((int) syscall(SYS_gettid), std::memory_order_release);
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET((tid - 1) % 15, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                uint32_t last_seq = 0;
                while (!stop_flag.load(std::memory_order_acquire)) {
                    uint32_t s = work_seq.load(std::memory_order_acquire);
                    if (s == last_seq || (s & 1u) == 0u) {
                        if (sleep_mode.load(std::memory_order_relaxed)) {
                            std::this_thread::sleep_for(std::chrono::microseconds(250));
                        } else {
                            _mm_pause();
                        }
                        continue;
                    }
                    if (stop_flag.load(std::memory_order_acquire) || cur_req == nullptr) break;
                    _mm_lfence();
                    last_seq = s;
                    if (s & 2u) {
                        exec_phase2(tid);
                    } else {
                        exec_phase1(tid);
                    }
                    done_count.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        }
        for (int tid = 1; tid < n_threads; ++tid) {
            while (worker_tids[tid].load(std::memory_order_acquire) < 0) _mm_pause();
        }
        sweep_aux_threads();
    }

    // Move all non-pool threads (CUDA driver helper threads, etc.) off the worker & host physical cores!
    void sweep_aux_threads() {
        cpu_set_t aux_set;
        CPU_ZERO(&aux_set);
        for (int c = n_threads - 1; c < 14; ++c) CPU_SET(c, &aux_set);
        for (int c = 16 + (n_threads - 1); c < 31; ++c) CPU_SET(c, &aux_set);
        const int s1_tid = s1_launcher_tid.load(std::memory_order_acquire);
        if (DIR* d = opendir("/proc/self/task")) {
            while (const dirent* e = readdir(d)) {
                if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
                const int tid = std::atoi(e->d_name);
                if (tid <= 0 || tid == host_tid || tid == s1_tid) continue;
                bool is_worker = false;
                for (int w = 1; w < n_threads; ++w) {
                    if (worker_tids[w].load(std::memory_order_relaxed) == tid) { is_worker = true; break; }
                }
                if (!is_worker) {
                    sched_setaffinity(tid, sizeof(cpu_set_t), &aux_set);
                }
            }
            closedir(d);
        }
    }

    ~CpuMoePool() {
        stop_flag.store(true, std::memory_order_release);
        // Do NOT increment `work_seq` here: turning an even idle sequence into an odd value
        // would cause workers that just passed `stop_flag` to execute a phantom phase on null/stale `cur_req`.
        for (auto& w : workers) if (w.joinable()) w.join();
    }

    // Unpack 1 NVFP4 64-weight block (`blk` = 36 bytes) and accumulate dot products against `cnt` tokens!
    __attribute__((target("avx2,fma")))
    inline void dot_blk64_multi(const uint8_t* __restrict__ blk, int cnt,
                                const signed char* __restrict__ xq_base, int xq_stride,
                                const float* __restrict__ xs_base, int xs_stride,
                                const int* __restrict__ tok_idx, __m128* __restrict__ acc4) const {
        const __m256i lut = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        const __m256i m0f = _mm256_set1_epi8(0x0F);
        const __m256i ones16 = _mm256_set1_epi16(1);

        const __m128 w_sc = _mm_setr_ps(ue_tab[blk[0]], ue_tab[blk[1]], ue_tab[blk[2]], ue_tab[blk[3]]);
        const __m256i raw = _mm256_loadu_si256((const __m256i*) (blk + 4));
        const __m256i w_lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw, m0f));
        const __m256i w_hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f));

        // w_02 holds [s0 (16B) | s2 (16B)], w_13 holds [s1 (16B) | s3 (16B)]
        const __m256i w_02 = _mm256_unpacklo_epi64(w_lo, w_hi);
        const __m256i w_13 = _mm256_unpackhi_epi64(w_lo, w_hi);
        const __m256i abs_02 = _mm256_abs_epi8(w_02);
        const __m256i abs_13 = _mm256_abs_epi8(w_13);

        for (int j = 0; j < cnt; ++j) {
            const int t = tok_idx[j];
            const signed char* xp = xq_base + (size_t) t * xq_stride;
            const __m256i x_02 = _mm256_loadu_si256((const __m256i*) xp);
            const __m256i x_13 = _mm256_loadu_si256((const __m256i*) (xp + 32));

            const __m256i p_02 = _mm256_madd_epi16(_mm256_maddubs_epi16(abs_02, _mm256_sign_epi8(x_02, w_02)), ones16);
            const __m256i p_13 = _mm256_madd_epi16(_mm256_maddubs_epi16(abs_13, _mm256_sign_epi8(x_13, w_13)), ones16);
            const __m256i h0 = _mm256_hadd_epi32(p_02, p_13);
            const __m128i dots4 = _mm_hadd_epi32(_mm256_castsi256_si128(h0), _mm256_extracti128_si256(h0, 1));

            const __m128 x_sc = _mm_loadu_ps(xs_base + (size_t) t * xs_stride);
            acc4[j] = _mm_fmadd_ps(_mm_cvtepi32_ps(dots4), _mm_mul_ps(w_sc, x_sc), acc4[j]);
        }
    }

    static inline float hsum_ps(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
    }

    // Phase 1: Gate + Up + SwiGLU + quantize `dq_perm` & `ds16`
    // Partitioned across `(cg, tile16)` where `NFF = 640` = `40` tiles of `16` rows (`40 * n_cold` tasks!)
    // Each 16-row tile is ALSO a complete 16-element quantization group, so each thread quantizes its own 16 rows
    // immediately into `dq_perm` and `ds16` in L1 cache!
    __attribute__((target("avx2,fma")))
    void exec_phase1(int tid) {
        const ColdLayerReq* req = cur_req;
        const int n_cold = req->n_cold;
        const int n_tiles = NFF / 16; // 40
        const int total_tasks = n_cold * n_tiles;

        for (int task = next_task1.fetch_add(1, std::memory_order_relaxed);
             task < total_tasks;
             task = next_task1.fetch_add(1, std::memory_order_relaxed)) {
            const int cg = task / n_tiles;
            const int tile = task - cg * n_tiles;
            const int r0 = tile * 16;
            const int cnt = req->cold_cnt[cg];
            const int gkey = req->cold_gkey[cg];
            const uint8_t* base = cur_pack + (size_t) gkey * PER;
            const float* tail = (const float*) (base + PER - 16);
            const float sc_gate = tail[0], sc_up = tail[1];

            int tok_idx[GRP_MAX];
            for (int j = 0; j < cnt; ++j) tok_idx[j] = req->cold_tok[cg][j];

            float act16[GRP_MAX][16];

            if (cnt == 1) {
                const int t0 = tok_idx[0];
                const signed char* xp_t = req->xq_perm + (size_t) t0 * NE;
                const float* xs_t = req->xs16 + (size_t) t0 * (NE / 16);
                const __m256i lut = _mm256_setr_epi8(
                    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
                const __m256i m0f = _mm256_set1_epi8(0x0F);
                const __m256i ones16 = _mm256_set1_epi16(1);
                float raw_g[16], raw_u[16];
                for (int dr = 0; dr < 16; ++dr) {
                    const int row = r0 + dr;
                    const uint8_t* g_ptr = base + (size_t) row * GU_ROW;
                    const uint8_t* u_ptr = base + (size_t) (NFF + row) * GU_ROW;
                    __m128 ag0 = _mm_setzero_ps(), au0 = _mm_setzero_ps();
                    for (int b = 0; b < (NE / 64); ++b) {
                        const __m256i x_02 = _mm256_loadu_si256((const __m256i*) (xp_t + b * 64));
                        const __m256i x_13 = _mm256_loadu_si256((const __m256i*) (xp_t + b * 64 + 32));
                        const __m128 x_sc = _mm_loadu_ps(xs_t + b * 4);

                        const uint8_t* gb = g_ptr + b * 36;
                        const __m128 gw_sc = _mm_setr_ps(ue_tab[gb[0]], ue_tab[gb[1]], ue_tab[gb[2]], ue_tab[gb[3]]);
                        const __m256i graw = _mm256_loadu_si256((const __m256i*) (gb + 4));
                        const __m256i gw_lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(graw, m0f));
                        const __m256i gw_hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(graw, 4), m0f));
                        const __m256i gw_02 = _mm256_unpacklo_epi64(gw_lo, gw_hi);
                        const __m256i gw_13 = _mm256_unpackhi_epi64(gw_lo, gw_hi);
                        const __m256i gp_02 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(gw_02), _mm256_sign_epi8(x_02, gw_02)), ones16);
                        const __m256i gp_13 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(gw_13), _mm256_sign_epi8(x_13, gw_13)), ones16);
                        const __m256i gh0 = _mm256_hadd_epi32(gp_02, gp_13);
                        const __m128i gdots = _mm_hadd_epi32(_mm256_castsi256_si128(gh0), _mm256_extracti128_si256(gh0, 1));
                        ag0 = _mm_fmadd_ps(_mm_cvtepi32_ps(gdots), _mm_mul_ps(gw_sc, x_sc), ag0);

                        const uint8_t* ub = u_ptr + b * 36;
                        const __m128 uw_sc = _mm_setr_ps(ue_tab[ub[0]], ue_tab[ub[1]], ue_tab[ub[2]], ue_tab[ub[3]]);
                        const __m256i uraw = _mm256_loadu_si256((const __m256i*) (ub + 4));
                        const __m256i uw_lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(uraw, m0f));
                        const __m256i uw_hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(uraw, 4), m0f));
                        const __m256i uw_02 = _mm256_unpacklo_epi64(uw_lo, uw_hi);
                        const __m256i uw_13 = _mm256_unpackhi_epi64(uw_lo, uw_hi);
                        const __m256i up_02 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(uw_02), _mm256_sign_epi8(x_02, uw_02)), ones16);
                        const __m256i up_13 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(uw_13), _mm256_sign_epi8(x_13, uw_13)), ones16);
                        const __m256i uh0 = _mm256_hadd_epi32(up_02, up_13);
                        const __m128i udots = _mm_hadd_epi32(_mm256_castsi256_si128(uh0), _mm256_extracti128_si256(uh0, 1));
                        au0 = _mm_fmadd_ps(_mm_cvtepi32_ps(udots), _mm_mul_ps(uw_sc, x_sc), au0);
                    }
                    raw_g[dr] = hsum_ps(ag0) * sc_gate;
                    raw_u[dr] = hsum_ps(au0) * sc_up;
                }
                for (int dr = 0; dr < 16; ++dr) {
                    act16[0][dr] = (raw_g[dr] / (1.0f + std::exp(-raw_g[dr]))) * raw_u[dr];
                }
            } else {
                for (int dr = 0; dr < 16; ++dr) {
                    const int row = r0 + dr;
                    const uint8_t* g_ptr = base + (size_t) row * GU_ROW;
                    const uint8_t* u_ptr = base + (size_t) (NFF + row) * GU_ROW;

                    __m128 ag4[GRP_MAX], au4[GRP_MAX];
                    for (int j = 0; j < cnt; ++j) {
                        ag4[j] = _mm_setzero_ps();
                        au4[j] = _mm_setzero_ps();
                    }

                    for (int b = 0; b < (NE / 64); ++b) {
                        dot_blk64_multi(g_ptr + b * 36, cnt, req->xq_perm + b * 64, NE, req->xs16 + b * 4, NE / 16, tok_idx, ag4);
                        dot_blk64_multi(u_ptr + b * 36, cnt, req->xq_perm + b * 64, NE, req->xs16 + b * 4, NE / 16, tok_idx, au4);
                    }

                    for (int j = 0; j < cnt; ++j) {
                        const float ag = hsum_ps(ag4[j]) * sc_gate;
                        const float au = hsum_ps(au4[j]) * sc_up;
                        act16[j][dr] = (ag / (1.0f + std::exp(-ag))) * au;
                    }
                }
            }

            // Quantize the 16-element group `tile` for each token `j < cnt` and write into pre-permuted `dq_perm`!
            // Within each 64-element block (`tile / 4`), sub-blocks `[0, 1, 2, 3]` are stored at `[0, 2, 1, 3]`!
            const int blk64 = tile >> 2;
            const int sub = tile & 3;
            const int perm_sub = (sub == 1) ? 2 : ((sub == 2) ? 1 : sub);
            const int dst_off = blk64 * 64 + perm_sub * 16;

            for (int j = 0; j < cnt; ++j) {
                float amax = 0.f;
                for (int dr = 0; dr < 16; ++dr) {
                    const float a = std::fabs(act16[j][dr]);
                    if (a > amax) amax = a;
                }
                const float d = amax / 127.0f;
                const float inv_d = d > 0.f ? (127.0f / amax) : 0.f;
                ds16[cg][j][tile] = d;
                signed char* qdst = dq_perm[cg][j] + dst_off;
                for (int dr = 0; dr < 16; ++dr) {
                    int v = _mm_cvtss_si32(_mm_set_ss(act16[j][dr] * inv_d));
                    if (v < -127) v = -127;
                    if (v > 127) v = 127;
                    qdst[dr] = (signed char) v;
                }
            }
        }
    }

    // Phase 2: Down projection (`NE = 2560` rows) + weighted accumulation into `cur_out [T, NE]`!
    // Partitioned by output row tile (`64` tiles of `40` rows = `2560` rows), so each thread owns disjoint columns
    // of `cur_out [t, row]` -> ZERO atomics and ZERO cross-thread reduction!
    __attribute__((target("avx2,fma")))
    void exec_phase2(int tid) {
        const ColdLayerReq* req = cur_req;
        const int n_cold = req->n_cold;
        const int T = req->T;
        const int tile_rows = 64;
        const int n_tiles = NE / tile_rows; // 40 tiles of 64 rows (256-byte cache-line aligned!)

        int seq_idx[GRP_MAX];
        for (int j = 0; j < GRP_MAX; ++j) seq_idx[j] = j;

        for (int tile = next_task2.fetch_add(1, std::memory_order_relaxed);
             tile < n_tiles;
             tile = next_task2.fetch_add(1, std::memory_order_relaxed)) {
            const int r0 = tile * tile_rows;
            for (int t = 0; t < T; ++t) {
                float* dst = cur_out + (size_t) t * NE + r0;
                for (int r = 0; r < tile_rows; r += 8) _mm256_storeu_ps(dst + r, _mm256_setzero_ps());
            }

            // Accumulate all `n_cold` experts for rows `[r0 .. r0 + tile_rows)`
            for (int cg = 0; cg < n_cold; ++cg) {
                const int cnt = req->cold_cnt[cg];
                const int gkey = req->cold_gkey[cg];
                const uint8_t* base = cur_pack + (size_t) gkey * PER;
                const float sc_down = ((const float*) (base + PER - 16))[2];
                const uint8_t* base_d = base + 2ull * NFF * GU_ROW;

                const signed char* xq_cg = &dq_perm[cg][0][0];
                const float* xs_cg = &ds16[cg][0][0];

                if (cnt == 1) {
                    const int t0 = req->cold_tok[cg][0];
                    const float wt_sc = req->cold_wt[cg][0] * sc_down;
                    float* out_t0 = cur_out + (size_t) t0 * NE;
                    const __m256i lut = _mm256_setr_epi8(
                        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
                    const __m256i m0f = _mm256_set1_epi8(0x0F);
                    const __m256i ones16 = _mm256_set1_epi16(1);
                    for (int dr = 0; dr < tile_rows; dr += 4) {
                        const int row = r0 + dr;
                        const uint8_t* rp0 = base_d + (size_t) (row + 0) * D_ROW;
                        const uint8_t* rp1 = base_d + (size_t) (row + 1) * D_ROW;
                        const uint8_t* rp2 = base_d + (size_t) (row + 2) * D_ROW;
                        const uint8_t* rp3 = base_d + (size_t) (row + 3) * D_ROW;
                        __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps(), a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
                        for (int b = 0; b < (NFF / 64); ++b) {
                            const __m256i x_02 = _mm256_loadu_si256((const __m256i*) (xq_cg + b * 64));
                            const __m256i x_13 = _mm256_loadu_si256((const __m256i*) (xq_cg + b * 64 + 32));
                            const __m128 x_sc = _mm_loadu_ps(xs_cg + b * 4);
                            auto step1 = [&](const uint8_t* blk, __m128& acc) {
                                const __m128 w_sc = _mm_setr_ps(ue_tab[blk[0]], ue_tab[blk[1]], ue_tab[blk[2]], ue_tab[blk[3]]);
                                const __m256i raw = _mm256_loadu_si256((const __m256i*) (blk + 4));
                                const __m256i w_lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw, m0f));
                                const __m256i w_hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f));
                                const __m256i w_02 = _mm256_unpacklo_epi64(w_lo, w_hi);
                                const __m256i w_13 = _mm256_unpackhi_epi64(w_lo, w_hi);
                                const __m256i p_02 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w_02), _mm256_sign_epi8(x_02, w_02)), ones16);
                                const __m256i p_13 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w_13), _mm256_sign_epi8(x_13, w_13)), ones16);
                                const __m256i h0 = _mm256_hadd_epi32(p_02, p_13);
                                const __m128i d4 = _mm_hadd_epi32(_mm256_castsi256_si128(h0), _mm256_extracti128_si256(h0, 1));
                                acc = _mm_fmadd_ps(_mm_cvtepi32_ps(d4), _mm_mul_ps(w_sc, x_sc), acc);
                            };
                            step1(rp0 + b * 36, a0);
                            step1(rp1 + b * 36, a1);
                            step1(rp2 + b * 36, a2);
                            step1(rp3 + b * 36, a3);
                        }
                        out_t0[row + 0] += wt_sc * hsum_ps(a0);
                        out_t0[row + 1] += wt_sc * hsum_ps(a1);
                        out_t0[row + 2] += wt_sc * hsum_ps(a2);
                        out_t0[row + 3] += wt_sc * hsum_ps(a3);
                    }
                } else {
                    for (int dr = 0; dr < tile_rows; ++dr) {
                        const int row = r0 + dr;
                        const uint8_t* r_ptr = base_d + (size_t) row * D_ROW;

                        __m128 acc4[GRP_MAX];
                        for (int j = 0; j < cnt; ++j) acc4[j] = _mm_setzero_ps();

                        for (int b = 0; b < (NFF / 64); ++b) {
                            dot_blk64_multi(r_ptr + b * 36, cnt, xq_cg + b * 64, NFF, xs_cg + b * 4, NFF / 16, seq_idx, acc4);
                        }

                        for (int j = 0; j < cnt; ++j) {
                            const int t = req->cold_tok[cg][j];
                            const float wt = req->cold_wt[cg][j];
                            cur_out[(size_t) t * NE + row] += wt * (hsum_ps(acc4[j]) * sc_down);
                        }
                    }
                }
            }
        }
    }

    void run(const ColdLayerReq* req, const uint8_t* host_pack, float* out_buf) {
        cur_req = req;
        cur_pack = host_pack;
        cur_out = out_buf;

        // Dispatch Phase 1 (seq & 1 == 1, seq & 2 == 0)
        next_task1.store(0, std::memory_order_relaxed);
        done_count.store(0, std::memory_order_relaxed);
        uint32_t prev_s = work_seq.load(std::memory_order_relaxed) & ~3u;
        if (prev_s >= 0xFFFFFFF0u) prev_s = 0u;
        uint32_t s1 = prev_s + 1u;
        work_seq.store(s1, std::memory_order_release);
        exec_phase1(0);
        while (done_count.load(std::memory_order_acquire) < n_threads - 1) _mm_pause();

        // Dispatch Phase 2 (seq & 1 == 1, seq & 2 == 2)
        next_task2.store(0, std::memory_order_relaxed);
        done_count.store(0, std::memory_order_relaxed);
        uint32_t s2 = s1 + 2u;
        work_seq.store(s2, std::memory_order_release);
        exec_phase2(0);
        while (done_count.load(std::memory_order_acquire) < n_threads - 1) _mm_pause();

        work_seq.store(s2 + 1u, std::memory_order_release); // back to even (idle)
    }
};
