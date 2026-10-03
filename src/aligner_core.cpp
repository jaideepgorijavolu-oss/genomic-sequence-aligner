#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <atomic>

// SSE2 is part of the x86-64 baseline (always on for MSVC x64, GCC and Clang x86-64).
// Other targets (e.g. ARM) use the scalar kernel.
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define ALIGNER_HAVE_SSE2 1
#else
#define ALIGNER_HAVE_SSE2 0
#endif

namespace py = pybind11;

// Large negative sentinel to avoid arithmetic underflow during additions
constexpr int NEG_INF = -100000000;

// Row-major 1D indexing helper
inline size_t idx(size_t i, size_t j, size_t stride) {
    return i * stride + j;
}

// ============================================================================
// 1. STANDARD NEEDLEMAN-WUNSCH & SMITH-WATERMAN (Linear Gap Penalties)
// ============================================================================

std::tuple<std::string, std::string, int> needleman_wunsch_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    size_t m = seq1.length();
    size_t n = seq2.length();
    size_t stride = n + 1;

    std::vector<int> dp((m + 1) * (n + 1), 0);

    for (size_t i = 0; i <= m; ++i) dp[idx(i, 0, stride)] = static_cast<int>(i) * gap_penalty;
    for (size_t j = 0; j <= n; ++j) dp[idx(0, j, stride)] = static_cast<int>(j) * gap_penalty;

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int score_diag = dp[idx(i - 1, j - 1, stride)] +
                             (seq1[i - 1] == seq2[j - 1] ? match_score : mismatch_penalty);
            int score_up = dp[idx(i - 1, j, stride)] + gap_penalty;
            int score_left = dp[idx(i, j - 1, stride)] + gap_penalty;
            dp[idx(i, j, stride)] = std::max({score_diag, score_up, score_left});
        }
    }

    std::string aligned1 = "";
    std::string aligned2 = "";
    size_t i = m, j = n;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 &&
            dp[idx(i, j, stride)] == dp[idx(i - 1, j - 1, stride)] +
            (seq1[i - 1] == seq2[j - 1] ? match_score : mismatch_penalty)) {
            aligned1 += seq1[i - 1];
            aligned2 += seq2[j - 1];
            --i;
            --j;
        } else if (i > 0 && dp[idx(i, j, stride)] == dp[idx(i - 1, j, stride)] + gap_penalty) {
            aligned1 += seq1[i - 1];
            aligned2 += '-';
            --i;
        } else {
            aligned1 += '-';
            aligned2 += seq2[j - 1];
            --j;
        }
    }

    std::reverse(aligned1.begin(), aligned1.end());
    std::reverse(aligned2.begin(), aligned2.end());

    return std::make_tuple(aligned1, aligned2, dp[idx(m, n, stride)]);
}

std::tuple<std::string, std::string, int> smith_waterman_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    size_t m = seq1.length();
    size_t n = seq2.length();
    size_t stride = n + 1;

    std::vector<int> dp((m + 1) * (n + 1), 0);

    int max_score = 0;
    size_t max_i = 0, max_j = 0;

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int score_diag = dp[idx(i - 1, j - 1, stride)] +
                             (seq1[i - 1] == seq2[j - 1] ? match_score : mismatch_penalty);
            int score_up = dp[idx(i - 1, j, stride)] + gap_penalty;
            int score_left = dp[idx(i, j - 1, stride)] + gap_penalty;

            int cell_val = std::max({0, score_diag, score_up, score_left});
            dp[idx(i, j, stride)] = cell_val;

            if (cell_val > max_score) {
                max_score = cell_val;
                max_i = i;
                max_j = j;
            }
        }
    }

    std::string aligned1 = "";
    std::string aligned2 = "";
    size_t i = max_i, j = max_j;

    while (i > 0 && j > 0 && dp[idx(i, j, stride)] > 0) {
        int current_score = dp[idx(i, j, stride)];
        int diag_score = dp[idx(i - 1, j - 1, stride)] +
                         (seq1[i - 1] == seq2[j - 1] ? match_score : mismatch_penalty);
        int up_score = dp[idx(i - 1, j, stride)] + gap_penalty;

        if (current_score == diag_score) {
            aligned1 += seq1[i - 1];
            aligned2 += seq2[j - 1];
            --i;
            --j;
        } else if (current_score == up_score) {
            aligned1 += seq1[i - 1];
            aligned2 += '-';
            --i;
        } else {
            aligned1 += '-';
            aligned2 += seq2[j - 1];
            --j;
        }
    }

    std::reverse(aligned1.begin(), aligned1.end());
    std::reverse(aligned2.begin(), aligned2.end());

    return std::make_tuple(aligned1, aligned2, max_score);
}

// ============================================================================
// 2. GOTOH ALGORITHM (Affine Gap Penalties: 3-State DP)
// ============================================================================

enum State { STATE_M, STATE_IX, STATE_IY };

std::tuple<std::string, std::string, int> gotoh_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_open,
    int gap_extend
) {
    size_t m = seq1.length();
    size_t n = seq2.length();
    size_t stride = n + 1;

    std::vector<int> M((m + 1) * (n + 1), NEG_INF);
    std::vector<int> Ix((m + 1) * (n + 1), NEG_INF);
    std::vector<int> Iy((m + 1) * (n + 1), NEG_INF);

    M[idx(0, 0, stride)] = 0;

    for (size_t i = 1; i <= m; ++i) {
        Ix[idx(i, 0, stride)] = gap_open + static_cast<int>(i) * gap_extend;
    }
    for (size_t j = 1; j <= n; ++j) {
        Iy[idx(0, j, stride)] = gap_open + static_cast<int>(j) * gap_extend;
    }

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int prev_best = std::max({M[idx(i - 1, j - 1, stride)],
                                      Ix[idx(i - 1, j - 1, stride)],
                                      Iy[idx(i - 1, j - 1, stride)]});
            int score = (seq1[i - 1] == seq2[j - 1]) ? match_score : mismatch_penalty;
            M[idx(i, j, stride)] = prev_best + score;

            // A gap may follow a gap in the other sequence (opening a new gap), as in
            // Biopython's PairwiseAligner; without this, e.g. "A" vs "T" with a large
            // mismatch penalty cannot take the cheaper "A-"/"-T" alignment.
            Ix[idx(i, j, stride)] = std::max({
                M[idx(i - 1, j, stride)] + gap_open + gap_extend,
                Ix[idx(i - 1, j, stride)] + gap_extend,
                Iy[idx(i - 1, j, stride)] + gap_open + gap_extend
            });

            Iy[idx(i, j, stride)] = std::max({
                M[idx(i, j - 1, stride)] + gap_open + gap_extend,
                Iy[idx(i, j - 1, stride)] + gap_extend,
                Ix[idx(i, j - 1, stride)] + gap_open + gap_extend
            });
        }
    }

    int best_score = std::max({M[idx(m, n, stride)], Ix[idx(m, n, stride)], Iy[idx(m, n, stride)]});

    State current_state = STATE_M;
    if (best_score == Ix[idx(m, n, stride)]) current_state = STATE_IX;
    else if (best_score == Iy[idx(m, n, stride)]) current_state = STATE_IY;

    std::string aligned1 = "";
    std::string aligned2 = "";
    size_t i = m, j = n;

    while (i > 0 || j > 0) {
        if (current_state == STATE_M) {
            aligned1 += seq1[i - 1];
            aligned2 += seq2[j - 1];
            int prev_best = std::max({M[idx(i - 1, j - 1, stride)],
                                      Ix[idx(i - 1, j - 1, stride)],
                                      Iy[idx(i - 1, j - 1, stride)]});
            if (prev_best == M[idx(i - 1, j - 1, stride)]) current_state = STATE_M;
            else if (prev_best == Ix[idx(i - 1, j - 1, stride)]) current_state = STATE_IX;
            else current_state = STATE_IY;
            --i;
            --j;
        } else if (current_state == STATE_IX) {
            aligned1 += seq1[i - 1];
            aligned2 += '-';
            const int here = Ix[idx(i, j, stride)];
            if (here == M[idx(i - 1, j, stride)] + gap_open + gap_extend) {
                current_state = STATE_M;
            } else if (here == Ix[idx(i - 1, j, stride)] + gap_extend) {
                current_state = STATE_IX;
            } else {
                current_state = STATE_IY;
            }
            --i;
        } else {
            aligned1 += '-';
            aligned2 += seq2[j - 1];
            const int here = Iy[idx(i, j, stride)];
            if (here == M[idx(i, j - 1, stride)] + gap_open + gap_extend) {
                current_state = STATE_M;
            } else if (here == Iy[idx(i, j - 1, stride)] + gap_extend) {
                current_state = STATE_IY;
            } else {
                current_state = STATE_IX;
            }
            --j;
        }
    }

    std::reverse(aligned1.begin(), aligned1.end());
    std::reverse(aligned2.begin(), aligned2.end());

    return std::make_tuple(aligned1, aligned2, best_score);
}

// ============================================================================
// 3. HIRSCHBERG ALGORITHM (Linear Space O(min(m, n)) Global Alignment)
// ============================================================================

std::vector<int> nw_score_row(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    size_t n = seq2.length();
    std::vector<int> prev(n + 1, 0);
    std::vector<int> curr(n + 1, 0);

    for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j) * gap_penalty;

    for (size_t i = 1; i <= seq1.length(); ++i) {
        curr[0] = static_cast<int>(i) * gap_penalty;
        for (size_t j = 1; j <= n; ++j) {
            int score_diag = prev[j - 1] + (seq1[i - 1] == seq2[j - 1] ? match_score : mismatch_penalty);
            int score_up = prev[j] + gap_penalty;
            int score_left = curr[j - 1] + gap_penalty;
            curr[j] = std::max({score_diag, score_up, score_left});
        }
        prev = curr;
    }
    return prev;
}

std::pair<std::string, std::string> hirschberg_recursive(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    size_t m = seq1.length();
    size_t n = seq2.length();

    if (m == 0) {
        return {std::string(n, '-'), seq2};
    }
    if (n == 0) {
        return {seq1, std::string(m, '-')};
    }
    if (m == 1 || n == 1) {
        auto res = needleman_wunsch_cpp(seq1, seq2, match_score, mismatch_penalty, gap_penalty);
        return {std::get<0>(res), std::get<1>(res)};
    }

    size_t mid = m / 2;
    std::string seq1_left = seq1.substr(0, mid);
    std::string seq1_right = seq1.substr(mid);

    std::string seq1_right_rev = seq1_right;
    std::string seq2_rev = seq2;
    std::reverse(seq1_right_rev.begin(), seq1_right_rev.end());
    std::reverse(seq2_rev.begin(), seq2_rev.end());

    std::vector<int> score_left = nw_score_row(seq1_left, seq2, match_score, mismatch_penalty, gap_penalty);
    std::vector<int> score_right = nw_score_row(seq1_right_rev, seq2_rev, match_score, mismatch_penalty, gap_penalty);

    size_t split_j = 0;
    int max_sum = NEG_INF;
    for (size_t j = 0; j <= n; ++j) {
        int sum = score_left[j] + score_right[n - j];
        if (sum > max_sum) {
            max_sum = sum;
            split_j = j;
        }
    }

    auto left_align = hirschberg_recursive(seq1_left, seq2.substr(0, split_j), match_score, mismatch_penalty, gap_penalty);
    auto right_align = hirschberg_recursive(seq1_right, seq2.substr(split_j), match_score, mismatch_penalty, gap_penalty);

    return {left_align.first + right_align.first, left_align.second + right_align.second};
}

std::tuple<std::string, std::string, int> hirschberg_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    // DP rows span seq2, so recurse with the shorter sequence second: O(min(m, n)) DP memory.
    const bool swap = seq2.length() > seq1.length();
    auto alignment = swap
        ? hirschberg_recursive(seq2, seq1, match_score, mismatch_penalty, gap_penalty)
        : hirschberg_recursive(seq1, seq2, match_score, mismatch_penalty, gap_penalty);
    if (swap) std::swap(alignment.first, alignment.second);

    // Compute final alignment score
    int score = 0;
    for (size_t k = 0; k < alignment.first.length(); ++k) {
        char c1 = alignment.first[k];
        char c2 = alignment.second[k];
        if (c1 == '-' || c2 == '-') score += gap_penalty;
        else if (c1 == c2) score += match_score;
        else score += mismatch_penalty;
    }

    return std::make_tuple(alignment.first, alignment.second, score);
}

// Score-only global alignment in two rows of length min(m, n) + 1 (no traceback).
int nw_score_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    const bool swap = seq2.length() > seq1.length();
    return swap ? nw_score_row(seq2, seq1, match_score, mismatch_penalty, gap_penalty).back()
                : nw_score_row(seq1, seq2, match_score, mismatch_penalty, gap_penalty).back();
}

// ============================================================================
// 4. LOCAL SEARCH: STRIPED SIMD SMITH-WATERMAN (Farrar 2007), AFFINE GAPS
// ============================================================================
//
// Score-only local alignment of one query against many targets, the core of database
// search (BLAST/SSEARCH-style). Scoring is a full 256x256 substitution table, so the
// same kernel handles match/mismatch DNA scoring and protein matrices such as BLOSUM62.
//
// Gap convention (same as gotoh_cpp): a gap of length k scores gap_open + k * gap_extend,
// so the first gap character costs open_cost = -(gap_open + gap_extend) and every further
// character costs ext_cost = -gap_extend.
//
// SIMD layout: the query is split into 8 interleaved "stripes" held in the 8 int16 lanes
// of an SSE2 register (query position i lives in lane i / seg_len, segment i % seg_len).
// Dependencies along the query (vertical gaps, F) are resolved with Farrar's lazy-F loop,
// which almost always exits after one pass. Scores are 16-bit saturating; if a score gets
// close to the int16 limit the target is rescored with the exact 32-bit scalar kernel.

class LocalScorer {
public:
    LocalScorer(const std::string& query, std::vector<int> table, int gap_open, int gap_extend)
        : query_(query), table_(std::move(table)),
          open_cost_(-(gap_open + gap_extend)), ext_cost_(-gap_extend) {
        if (table_.size() != 256 * 256) {
            throw std::invalid_argument("substitution table must have 256*256 entries");
        }
        if (gap_extend > 0 || gap_open > 0) {
            throw std::invalid_argument("gap_open and gap_extend must be <= 0");
        }
        max_abs_score_ = 0;
        for (int v : table_) max_abs_score_ = std::max(max_abs_score_, std::abs(v));
        build_profile();
    }

    const std::string& query() const { return query_; }

    bool simd_enabled() const { return simd_; }

    // Exact reference kernel: O(m) memory, 32-bit scores.
    int score_scalar(const std::string& target) const {
        const size_t m = query_.size();
        if (m == 0 || target.empty()) return 0;
        std::vector<int> H(m + 1, 0), E(m + 1, 0);
        int best = 0;
        for (unsigned char t : target) {
            const int* row = &table_[static_cast<size_t>(t)];
            int diag = 0, h_up = 0, f = 0;
            for (size_t i = 1; i <= m; ++i) {
                // E: gap consuming target characters (horizontal); F: consuming query (vertical).
                const int e = std::max(E[i] - ext_cost_, H[i] - open_cost_);
                f = std::max(f - ext_cost_, h_up - open_cost_);
                const int sub = row[static_cast<size_t>(static_cast<unsigned char>(query_[i - 1])) * 256];
                const int h = std::max({0, diag + sub, e, f});
                diag = H[i];
                H[i] = h;
                E[i] = e;
                h_up = h;
                if (h > best) best = h;
            }
        }
        return best;
    }

    int score(const std::string& target) const {
#if ALIGNER_HAVE_SSE2
        if (simd_) {
            const int s = score_sse2(target);
            // Saturation guard: near INT16_MAX the 16-bit result may be clipped.
            if (s < INT16_MAX - max_abs_score_ - 1) return s;
        }
#endif
        return score_scalar(target);
    }

    std::vector<int> score_many(const std::vector<std::string>& targets, int threads) const {
        std::vector<int> out(targets.size(), 0);
        unsigned n = threads > 0 ? static_cast<unsigned>(threads) : std::thread::hardware_concurrency();
        n = std::max(1u, std::min<unsigned>(n, static_cast<unsigned>(targets.size())));
        // Dynamic scheduling: targets differ in length, so threads pull small chunks.
        std::atomic<size_t> next{0};
        const size_t chunk = 16;
        auto worker = [&]() {
            for (;;) {
                const size_t start = next.fetch_add(chunk);
                if (start >= targets.size()) return;
                const size_t stop = std::min(targets.size(), start + chunk);
                for (size_t k = start; k < stop; ++k) out[k] = score(targets[k]);
            }
        };
        if (n == 1) {
            worker();
            return out;
        }
        std::vector<std::thread> pool;
        pool.reserve(n);
        for (unsigned t = 0; t < n; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        return out;
    }

private:
    std::string query_;
    std::vector<int> table_;
    int open_cost_, ext_cost_;
    int max_abs_score_ = 0;
    bool simd_ = false;
    size_t seg_len_ = 0;
#if ALIGNER_HAVE_SSE2
    std::vector<__m128i> profile_;  // [256 symbols][seg_len_] query profile
#endif

    void build_profile() {
#if ALIGNER_HAVE_SSE2
        const size_t m = query_.size();
        // int16 lanes: scores and gap costs must leave headroom below the saturation limit.
        simd_ = m > 0 && max_abs_score_ < 1000 && open_cost_ < 16000 && ext_cost_ <= open_cost_;
        if (!simd_) return;
        seg_len_ = (m + 7) / 8;
        profile_.resize(256 * seg_len_);
        for (size_t c = 0; c < 256; ++c) {
            for (size_t j = 0; j < seg_len_; ++j) {
                alignas(16) int16_t lanes[8];
                for (size_t k = 0; k < 8; ++k) {
                    const size_t i = k * seg_len_ + j;
                    // Padding past the query end can never raise the max.
                    lanes[k] = i < m
                        ? static_cast<int16_t>(table_[static_cast<size_t>(static_cast<unsigned char>(query_[i])) * 256 + c])
                        : INT16_MIN;
                }
                profile_[c * seg_len_ + j] = _mm_load_si128(reinterpret_cast<const __m128i*>(lanes));
            }
        }
#endif
    }

#if ALIGNER_HAVE_SSE2
    int score_sse2(const std::string& target) const {
        if (target.empty()) return 0;
        const size_t S = seg_len_;
        std::vector<__m128i> buf(3 * S, _mm_setzero_si128());
        __m128i* h_store = buf.data();
        __m128i* h_load = buf.data() + S;
        __m128i* e_vec = buf.data() + 2 * S;
        const __m128i v_open = _mm_set1_epi16(static_cast<int16_t>(open_cost_));
        const __m128i v_ext = _mm_set1_epi16(static_cast<int16_t>(ext_cost_));
        __m128i v_max = _mm_setzero_si128();

        // E and F start at 0 and only shrink via unsigned saturating subtraction, so they
        // stay >= 0 and max(H, E, F) already applies the local-alignment floor of 0.
        for (unsigned char t : target) {
            const __m128i* prof = &profile_[static_cast<size_t>(t) * S];
            __m128i v_f = _mm_setzero_si128();
            // Diagonal input for segment 0: previous column's last segment shifted one lane.
            __m128i v_h = _mm_slli_si128(h_store[S - 1], 2);
            std::swap(h_load, h_store);

            for (size_t j = 0; j < S; ++j) {
                v_h = _mm_adds_epi16(v_h, prof[j]);
                __m128i v_e = e_vec[j];
                v_h = _mm_max_epi16(v_h, v_e);
                v_h = _mm_max_epi16(v_h, v_f);
                v_max = _mm_max_epi16(v_max, v_h);
                h_store[j] = v_h;

                const __m128i v_h_open = _mm_subs_epu16(v_h, v_open);
                v_e = _mm_max_epi16(_mm_subs_epu16(v_e, v_ext), v_h_open);
                e_vec[j] = v_e;
                v_f = _mm_max_epi16(_mm_subs_epu16(v_f, v_ext), v_h_open);
                v_h = h_load[j];
            }

            // Lazy-F: carry vertical gaps across stripe boundaries until no H can improve.
            for (int pass = 0; pass < 8; ++pass) {
                v_f = _mm_slli_si128(v_f, 2);
                for (size_t j = 0; j < S; ++j) {
                    __m128i v_hj = h_store[j];
                    const __m128i v_hj_open = _mm_subs_epu16(v_hj, v_open);
                    if (!_mm_movemask_epi8(_mm_cmpgt_epi16(v_f, v_hj_open))) {
                        pass = 8;  // F cannot beat opening a fresh gap here or further down
                        break;
                    }
                    v_hj = _mm_max_epi16(v_hj, v_f);
                    h_store[j] = v_hj;
                    v_max = _mm_max_epi16(v_max, v_hj);
                    // Raised H also raises the next column's horizontal-gap state.
                    const __m128i v_new_open = _mm_subs_epu16(v_hj, v_open);
                    e_vec[j] = _mm_max_epi16(e_vec[j], v_new_open);
                    v_f = _mm_subs_epu16(v_f, v_ext);
                }
            }
        }

        alignas(16) int16_t lanes[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(lanes), v_max);
        return *std::max_element(lanes, lanes + 8);
    }
#endif
};

// ============================================================================
// PYBIND11 MODULE EXPORT
// ============================================================================

PYBIND11_MODULE(aligner_core, m) {
    m.doc() = "High-performance C++ Sequence Alignment DP Kernels";
    // Arguments are converted to std::string before the call, so the kernels can run
    // without the GIL and alignments in other Python threads proceed in parallel.
    using release_gil = py::call_guard<py::gil_scoped_release>;
    m.def("needleman_wunsch_cpp", &needleman_wunsch_cpp, "Needleman-Wunsch global alignment", release_gil());
    m.def("smith_waterman_cpp", &smith_waterman_cpp, "Smith-Waterman local alignment", release_gil());
    m.def("gotoh_cpp", &gotoh_cpp, "Gotoh affine gap penalty global alignment", release_gil());
    m.def("hirschberg_cpp", &hirschberg_cpp, "Hirschberg linear space global alignment", release_gil());
    m.def("nw_score_cpp", &nw_score_cpp, "Linear-space Needleman-Wunsch score (no traceback)", release_gil());

    m.attr("HAVE_SSE2") = py::bool_(ALIGNER_HAVE_SSE2 != 0);
    py::class_<LocalScorer>(m, "LocalScorer",
        "Striped SIMD Smith-Waterman scorer for one query against many targets")
        .def(py::init<const std::string&, std::vector<int>, int, int>(),
             py::arg("query"), py::arg("table"), py::arg("gap_open"), py::arg("gap_extend"))
        .def("score", &LocalScorer::score, py::arg("target"), release_gil(),
             "Best local alignment score (SIMD, exact fallback near int16 overflow)")
        .def("score_scalar", &LocalScorer::score_scalar, py::arg("target"), release_gil(),
             "Best local alignment score (scalar 32-bit reference kernel)")
        .def("score_many", &LocalScorer::score_many, py::arg("targets"), py::arg("threads") = 0, release_gil(),
             "Scores for every target, computed on `threads` worker threads (0 = all cores)")
        .def_property_readonly("simd_enabled", &LocalScorer::simd_enabled)
        .def_property_readonly("query", &LocalScorer::query);
}