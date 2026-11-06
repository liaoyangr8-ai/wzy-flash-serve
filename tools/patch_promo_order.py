#!/usr/bin/env python3
"""Deterministic promotion ordering (README 7.13).

Root cause of the run-to-run divergence (proven by the no-promo experiment, 9/9 identical runs):
`split_route_hit_cold_kernel` hands out cold-group indices via atomicAdd, so the SERVICE order of
cold groups is nondeterministic. `service_one_cold_layer` pushes eligible experts into
`pending_promo` in exactly that order, and the push loop stops as soon as the queue reaches max_q
(WZY_PROMO_QUEUE=16). Which experts sit in the queue when it fills is therefore a coin flip ->
different promoted subsets -> different VRAM-resident maps -> different accumulation order ->
different greedy token stream.

Fix, in two deterministic steps, no new VRAM traffic:
  1. service_one_cold_layer: push every ELIGIBLE gkey (drop the `size() < max_q` cutoff), so the
     pending set no longer depends on atomic order;
  2. promote_cold: sort pending_promo by descending frequency then gkey (a deterministic priority),
     apply the max_q cap AFTER the sort, and iterate. The promoted subset becomes a pure function
     of the window's routed groups and the freq map.
"""
import shutil

STG = "/home/liaoyang/wzy-flash-serve/src/wzy_stage_v2.cuh"

s = open(STG, encoding="utf-8", errors="surrogateescape", newline="").read()
shutil.copy2(STG, "/tmp/wzy_stage_v2.cuh.bak-promo")
s = s.replace("\r\n", "\n")


def sub_once(old, new, what):
    global s
    n = s.count(old)
    assert n == 1, "%s: anchor count %d (expected 1)\n%r" % (what, n, old[:200])
    s = s.replace(old, new)


# ---- 1. collect all eligible candidates; the queue cap moves to promote_cold ---------------------
sub_once(
    "                        const uint32_t f = confirmed_use ? (cold_freq[gk] += (uint32_t) cnt_cg) : cold_freq[gk];\n"
    "                        static const uint32_t env_min_f = std::getenv(\"WZY_PROMO_MIN_F\") ? (uint32_t) std::max(1, std::atoi(std::getenv(\"WZY_PROMO_MIN_F\"))) : 4u;\n"
    "                        static const int env_max_q = std::getenv(\"WZY_PROMO_QUEUE\") ? std::max(1, std::atoi(std::getenv(\"WZY_PROMO_QUEUE\"))) : 16;\n"
    "                        const uint32_t min_f = env_min_f;\n"
    "                        const int max_q = env_max_q;\n"
    "                        if (confirmed_use && f >= min_f && h_lut[gk] < 0 && (int) pending_promo.size() < max_q) {\n"
    "                            pending_promo.push_back(gk);\n"
    "                        }",
    "                        const uint32_t f = confirmed_use ? (cold_freq[gk] += (uint32_t) cnt_cg) : cold_freq[gk];\n"
    "                        static const uint32_t env_min_f = std::getenv(\"WZY_PROMO_MIN_F\") ? (uint32_t) std::max(1, std::atoi(std::getenv(\"WZY_PROMO_MIN_F\"))) : 4u;\n"
    "                        const uint32_t min_f = env_min_f;\n"
    "                        // README 7.13: push EVERY eligible candidate. The old `pending_promo.size() < max_q`\n"
    "                        // cutoff here made the queued subset depend on the kernel's nondeterministic\n"
    "                        // atomicAdd service order -- the root cause of irreproducible greedy decode.\n"
    "                        // The queue cap is applied deterministically in promote_cold after sorting.\n"
    "                        if (confirmed_use && f >= min_f && h_lut[gk] < 0) {\n"
    "                            pending_promo.push_back(gk);\n"
    "                        }",
    "collect-all-eligible")

# ---- 2. deterministic priority order, then the cap -----------------------------------------------
sub_once(
    "            CHECK(cudaSetDevice(dev));\n"
    "            int promoted = 0;\n"
    "            static const int env_max_promo = std::getenv(\"WZY_PROMO_PER_WIN\") ? std::max(1, std::atoi(std::getenv(\"WZY_PROMO_PER_WIN\"))) : 8;\n"
    "            static const int env_max_scan = std::getenv(\"WZY_PROMO_SCAN\") ? std::max(16, std::atoi(std::getenv(\"WZY_PROMO_SCAN\"))) : 256;\n"
    "            const int max_promo = env_max_promo;\n"
    "            const int max_scans = std::min(adapt_cap, env_max_scan);\n"
    "            for (int gkey : pending_promo) {",
    "            CHECK(cudaSetDevice(dev));\n"
    "            int promoted = 0;\n"
    "            static const int env_max_promo = std::getenv(\"WZY_PROMO_PER_WIN\") ? std::max(1, std::atoi(std::getenv(\"WZY_PROMO_PER_WIN\"))) : 8;\n"
    "            static const int env_max_scan = std::getenv(\"WZY_PROMO_SCAN\") ? std::max(16, std::atoi(std::getenv(\"WZY_PROMO_SCAN\"))) : 256;\n"
    "            const int max_promo = env_max_promo;\n"
    "            const int max_scans = std::min(adapt_cap, env_max_scan);\n"
    "            // README 7.13: deterministic priority. `pending_promo` is in nondeterministic atomicAdd\n"
    "            // service order; sorting by (freq desc, gkey asc) makes the capped subset a pure function\n"
    "            // of this window's routed groups. Same max_q budget as before (WZY_PROMO_QUEUE).\n"
    "            {\n"
    "                static const int env_max_q = std::getenv(\"WZY_PROMO_QUEUE\") ? std::max(1, std::atoi(std::getenv(\"WZY_PROMO_QUEUE\"))) : 16;\n"
    "                const int max_q = env_max_q;\n"
    "                std::sort(pending_promo.begin(), pending_promo.end(),\n"
    "                          [&](int a, int b) {\n"
    "                              const uint32_t fa = cold_freq[a], fb = cold_freq[b];\n"
    "                              if (fa != fb) return fa > fb;   // hottest first\n"
    "                              return a < b;                   // stable, value-independent tie-break\n"
    "                          });\n"
    "                if ((int) pending_promo.size() > max_q) pending_promo.resize(max_q);\n"
    "            }\n"
    "            for (int gkey : pending_promo) {",
    "deterministic sort + cap")

open(STG, "w", encoding="utf-8", errors="surrogateescape").write(s)
print("promo order patch applied")
