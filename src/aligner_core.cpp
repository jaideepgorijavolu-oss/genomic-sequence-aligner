#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <climits>

namespace py = pybind11;

// Large negative sentinel to avoid arithmetic underflow during additions
constexpr int NEG_INF = -1e8;

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

            Ix[idx(i, j, stride)] = std::max(
                M[idx(i - 1, j, stride)] + gap_open + gap_extend,
                Ix[idx(i - 1, j, stride)] + gap_extend
            );

            Iy[idx(i, j, stride)] = std::max(
                M[idx(i, j - 1, stride)] + gap_open + gap_extend,
                Iy[idx(i, j - 1, stride)] + gap_extend
            );
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
            if (Ix[idx(i, j, stride)] == M[idx(i - 1, j, stride)] + gap_open + gap_extend) {
                current_state = STATE_M;
            } else {
                current_state = STATE_IX;
            }
            --i;
        } else {
            aligned1 += '-';
            aligned2 += seq2[j - 1];
            if (Iy[idx(i, j, stride)] == M[idx(i, j - 1, stride)] + gap_open + gap_extend) {
                current_state = STATE_M;
            } else {
                current_state = STATE_IY;
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
    auto alignment = hirschberg_recursive(seq1, seq2, match_score, mismatch_penalty, gap_penalty);

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

// ============================================================================
// PYBIND11 MODULE EXPORT
// ============================================================================

PYBIND11_MODULE(aligner_core, m) {
    m.doc() = "High-performance C++ Sequence Alignment DP Kernels";
    m.def("needleman_wunsch_cpp", &needleman_wunsch_cpp, "Needleman-Wunsch global alignment");
    m.def("smith_waterman_cpp", &smith_waterman_cpp, "Smith-Waterman local alignment");
    m.def("gotoh_cpp", &gotoh_cpp, "Gotoh affine gap penalty global alignment");
    m.def("hirschberg_cpp", &hirschberg_cpp, "Hirschberg linear space global alignment");
}