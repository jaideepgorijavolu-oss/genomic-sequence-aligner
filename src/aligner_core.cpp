#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

inline int get_score(char a, char b, int match, int mismatch) {
    return (a == b) ? match : mismatch;
}

std::tuple<std::string, std::string, int> needleman_wunsch_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    int m = static_cast<int>(seq1.length());
    int n = static_cast<int>(seq2.length());
    int stride = n + 1;

    // Contiguous 1D vector for optimal cache performance
    std::vector<int> matrix((m + 1) * stride, 0);

    for (int i = 1; i <= m; ++i) {
        matrix[i * stride] = i * gap_penalty;
    }
    for (int j = 1; j <= n; ++j) {
        matrix[j] = j * gap_penalty;
    }

    // Dynamic programming fill
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int match = matrix[(i - 1) * stride + (j - 1)] + 
                        get_score(seq1[i - 1], seq2[j - 1], match_score, mismatch_penalty);
            int del   = matrix[(i - 1) * stride + j] + gap_penalty;
            int ins   = matrix[i * stride + (j - 1)] + gap_penalty;
            matrix[i * stride + j] = std::max({match, del, ins});
        }
    }

    // Traceback
    std::string aligned1, aligned2;
    aligned1.reserve(m + n);
    aligned2.reserve(m + n);

    int i = m, j = n;
    while (i > 0 && j > 0) {
        int current = matrix[i * stride + j];
        int diag    = matrix[(i - 1) * stride + (j - 1)];
        int up      = matrix[(i - 1) * stride + j];

        if (current == diag + get_score(seq1[i - 1], seq2[j - 1], match_score, mismatch_penalty)) {
            aligned1.push_back(seq1[i - 1]);
            aligned2.push_back(seq2[j - 1]);
            --i;
            --j;
        } else if (current == up + gap_penalty) {
            aligned1.push_back(seq1[i - 1]);
            aligned2.push_back('-');
            --i;
        } else {
            aligned1.push_back('-');
            aligned2.push_back(seq2[j - 1]);
            --j;
        }
    }

    while (i > 0) {
        aligned1.push_back(seq1[i - 1]);
        aligned2.push_back('-');
        --i;
    }
    while (j > 0) {
        aligned1.push_back('-');
        aligned2.push_back(seq2[j - 1]);
        --j;
    }

    std::reverse(aligned1.begin(), aligned1.end());
    std::reverse(aligned2.begin(), aligned2.end());

    return {aligned1, aligned2, matrix[m * stride + n]};
}

std::tuple<std::string, std::string, int> smith_waterman_cpp(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_penalty,
    int gap_penalty
) {
    int m = static_cast<int>(seq1.length());
    int n = static_cast<int>(seq2.length());
    int stride = n + 1;

    std::vector<int> matrix((m + 1) * stride, 0);

    int max_score = 0;
    int max_i = 0, max_j = 0;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int match = matrix[(i - 1) * stride + (j - 1)] + 
                        get_score(seq1[i - 1], seq2[j - 1], match_score, mismatch_penalty);
            int del   = matrix[(i - 1) * stride + j] + gap_penalty;
            int ins   = matrix[i * stride + (j - 1)] + gap_penalty;
            int score = std::max({0, match, del, ins});
            matrix[i * stride + j] = score;

            if (score > max_score) {
                max_score = score;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Traceback from maximum cell
    std::string aligned1, aligned2;
    aligned1.reserve(m + n);
    aligned2.reserve(m + n);

    int i = max_i, j = max_j;
    while (i > 0 && j > 0 && matrix[i * stride + j] > 0) {
        int current = matrix[i * stride + j];
        int diag    = matrix[(i - 1) * stride + (j - 1)];
        int up      = matrix[(i - 1) * stride + j];

        if (current == diag + get_score(seq1[i - 1], seq2[j - 1], match_score, mismatch_penalty)) {
            aligned1.push_back(seq1[i - 1]);
            aligned2.push_back(seq2[j - 1]);
            --i;
            --j;
        } else if (current == up + gap_penalty) {
            aligned1.push_back(seq1[i - 1]);
            aligned2.push_back('-');
            --i;
        } else {
            aligned1.push_back('-');
            aligned2.push_back(seq2[j - 1]);
            --j;
        }
    }

    std::reverse(aligned1.begin(), aligned1.end());
    std::reverse(aligned2.begin(), aligned2.end());

    return {aligned1, aligned2, max_score};
}

PYBIND11_MODULE(aligner_core, m) {
    m.doc() = "C++ accelerated sequence alignment core";
    m.def("needleman_wunsch_cpp", &needleman_wunsch_cpp, "C++ Needleman-Wunsch algorithm");
    m.def("smith_waterman_cpp", &smith_waterman_cpp, "C++ Smith-Waterman algorithm");
}