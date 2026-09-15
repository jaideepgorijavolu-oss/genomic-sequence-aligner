from typing import Tuple

try:
    import aligner_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


class SequenceAligner:
    """
    High-Performance Pairwise Sequence Alignment Engine.
    Provides Needleman-Wunsch, Smith-Waterman, Gotoh (Affine Gaps),
    and Hirschberg (Linear Space) DP algorithms.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_penalty: int = -1,
        gap_penalty: int = -2,
        gap_open: int = -3,
        gap_extend: int = -1,
        use_cpp: bool = True
    ):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.use_cpp = use_cpp and CPP_AVAILABLE

    def needleman_wunsch(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Global alignment using standard linear gap penalty."""
        if self.use_cpp:
            return aligner_core.needleman_wunsch_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._needleman_wunsch_py(seq1, seq2)

    def smith_waterman(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Local alignment for optimal motif/subsequence identification."""
        if self.use_cpp:
            return aligner_core.smith_waterman_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._smith_waterman_py(seq1, seq2)

    def gotoh(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Global alignment with Affine Gap Penalties (W = open + k * extend)."""
        if self.use_cpp:
            return aligner_core.gotoh_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_open, self.gap_extend
            )
        raise NotImplementedError("Pure Python fallback for Gotoh algorithm is not enabled.")

    def hirschberg(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Linear Space O(min(m, n)) Global Alignment."""
        if self.use_cpp:
            return aligner_core.hirschberg_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        raise NotImplementedError("Pure Python fallback for Hirschberg algorithm is not enabled.")

    # --- Pure Python Implementations ---

    def _needleman_wunsch_py(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i * self.gap_penalty
        for j in range(n + 1):
            dp[0][j] = j * self.gap_penalty

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = dp[i - 1][j - 1] + (self.match_score if seq1[i - 1] == seq2[j - 1] else self.mismatch_penalty)
                delete = dp[i - 1][j] + self.gap_penalty
                insert = dp[i][j - 1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)

        aligned1, aligned2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                self.match_score if seq1[i - 1] == seq2[j - 1] else self.mismatch_penalty
            ):
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + self.gap_penalty:
                aligned1.append(seq1[i - 1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j - 1])
                j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), dp[m][n]

    def _smith_waterman_py(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_score = 0
        max_pos = (0, 0)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = dp[i - 1][j - 1] + (self.match_score if seq1[i - 1] == seq2[j - 1] else self.mismatch_penalty)
                delete = dp[i - 1][j] + self.gap_penalty
                insert = dp[i][j - 1] + self.gap_penalty
                score = max(0, match, delete, insert)
                dp[i][j] = score
                if score > max_score:
                    max_score = score
                    max_pos = (i, j)

        aligned1, aligned2 = [], []
        i, j = max_pos
        while i > 0 and j > 0 and dp[i][j] > 0:
            match = dp[i - 1][j - 1] + (self.match_score if seq1[i - 1] == seq2[j - 1] else self.mismatch_penalty)
            delete = dp[i - 1][j] + self.gap_penalty
            if dp[i][j] == match:
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif dp[i][j] == delete:
                aligned1.append(seq1[i - 1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j - 1])
                j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), max_score