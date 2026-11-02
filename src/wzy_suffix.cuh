// wzy_suffix.cuh -- Zero-GPU-Cost 4-Way Trigram Suffix Drafter & Lookup-Chain Extension (exact port of Strata SuffixDrafter)
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct SuffixDrafter {
    static constexpr int WAYS = 4;

    struct Slot {
        uint64_t key = 0;
        uint32_t pos[WAYS] = {};
        uint8_t n = 0;
    };

    int min_match_ = 3;
    int max_match_ = 32;
    std::vector<int> hist_;
    std::vector<Slot> table_;
    size_t mask_ = 0;
    int last_match_ = 0;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19)
        : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
        size_t cap = 1;
        while (cap < capacity_tokens * 2) cap <<= 1;
        table_.assign(cap, Slot{});
        mask_ = cap - 1;
        hist_.reserve(capacity_tokens);
    }

    static inline uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        return x ^ (x >> 33);
    }

    inline uint64_t key_at(size_t end) const {
        const uint64_t a = (uint32_t) hist_[end - 2];
        const uint64_t b = (uint32_t) hist_[end - 1];
        const uint64_t c = (uint32_t) hist_[end];
        return (mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1))) | 1ull;
    }

    inline Slot* find_slot(uint64_t key, bool insert) {
        for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
            Slot& s = table_[i];
            if (s.key == key) return &s;
            if (s.key == 0) {
                if (!insert) return nullptr;
                s.key = key;
                return &s;
            }
        }
        return nullptr;
    }

    inline const Slot* find_slot_const(uint64_t key) const {
        for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
            const Slot& s = table_[i];
            if (s.key == key) return &s;
            if (s.key == 0) return nullptr;
        }
        return nullptr;
    }

    void reset() {
        hist_.clear();
        std::fill(table_.begin(), table_.end(), Slot{});
        last_match_ = 0;
    }

    void append(const int* tokens, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            hist_.push_back(tokens[i]);
            const size_t end = hist_.size() - 1;
            if (end < 2) continue;
            Slot* s = find_slot(key_at(end), true);
            if (s == nullptr) continue;
            for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
            s->pos[0] = (uint32_t) end;
            if (s->n < WAYS) ++s->n;
        }
    }

    void append_one(int tok) { append(&tok, 1); }
    int last_match() const { return last_match_; }

    int propose(int max_k, int* out) {
        last_match_ = 0;
        const size_t n = hist_.size();
        if (n < 4 || max_k <= 0) return 0;
        const size_t cur = n - 1;
        const Slot* s = find_slot_const(key_at(cur));
        if (s == nullptr) return 0;
        size_t best_end = 0;
        int best_len = 0;
        for (int w = 0; w < s->n; ++w) {
            const size_t p = s->pos[w];
            if (p + 32 >= cur) continue;
            int len = 0;
            while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
            if (len > best_len) {
                best_len = len;
                best_end = p;
            }
        }
        if (best_len < min_match_) return 0;
        last_match_ = best_len;
        int k = 0;
        for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
        return k;
    }

    int propose_after(const int* extra, int n_extra, int max_k, int* out) {
        if (n_extra <= 0) return propose(max_k, out);
        const size_t n0 = hist_.size();
        hist_.insert(hist_.end(), extra, extra + n_extra);
        const int k = propose(max_k, out);
        hist_.resize(n0);
        return k;
    }
};
