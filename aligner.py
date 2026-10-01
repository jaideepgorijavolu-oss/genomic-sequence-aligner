from typing import NamedTuple, Tuple

try:
    import aligner_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

NEG_INF = -10**8  # same sentinel as the C++ core

# Full-matrix methods (NW, SW, Gotoh) allocate O(m*n) ints. Past this many cells,
# refuse and point at Hirschberg instead of exhausting RAM.
DEFAULT_MAX_CELLS = 100_000_000


class AlignmentStats(NamedTuple):
    length: int
    matches: int
    mismatches: int
    gaps: int
    gap_opens: int
    identity: float  # matches / alignment length


def alignment_stats(aligned1: str, aligned2: str) -> AlignmentStats:
    if len(aligned1) != len(aligned2):
        raise ValueError("aligned sequences must have equal length")
    matches = mismatches = gaps = gap_opens = 0
    prev_gap = None
    for c1, c2 in zip(aligned1, aligned2):
        if c1 == "-" or c2 == "-":
            gaps += 1
            side = 1 if c1 == "-" else 2
            if prev_gap != side:
                gap_opens += 1
            prev_gap = side
        else:
            prev_gap = None
            if c1 == c2:
                matches += 1
            else:
                mismatches += 1
    n = len(aligned1)
    return AlignmentStats(n, matches, mismatches, gaps, gap_opens, matches / n if n else 0.0)


def format_alignment(aligned1: str, aligned2: str, width: int = 60) -> str:
    """BLAST-style block view: '|' match, '.' mismatch, ' ' gap."""
    mid = "".join(
        " " if a == "-" or b == "-" else ("|" if a == b else ".") for a, b in zip(aligned1, aligned2)
    )
    blocks = []
    for k in range(0, len(aligned1), width):
        blocks.append("\n".join((aligned1[k:k + width], mid[k:k + width], aligned2[k:k + width])))
    return "\n\n".join(blocks)


class SequenceAligner:
    """
    High-Performance Pairwise Sequence Alignment Engine.
    Provides Needleman-Wunsch, Smith-Waterman, Gotoh (Affine Gaps),
    and Hirschberg (Linear Space) DP algorithms, each with a C++ kernel
    and a pure-Python reference implementation that produce identical results.

    Gap scoring: linear methods charge `gap_penalty` per gap character; Gotoh charges
    `gap_open + k * gap_extend` for a gap of length k.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_penalty: int = -1,
        gap_penalty: int = -2,
        gap_open: int = -3,
        gap_extend: int = -1,
        use_cpp: bool = True,
        case_sensitive: bool = False,
        max_cells: int = DEFAULT_MAX_CELLS,
    ):
        for name, value in (("match_score", match_score), ("mismatch_penalty", mismatch_penalty),
                            ("gap_penalty", gap_penalty), ("gap_open", gap_open), ("gap_extend", gap_extend)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self.case_sensitive = case_sensitive
        self.max_cells = max_cells

    @property
    def backend(self) -> str:
        return "cpp" if self.use_cpp else "python"

    # --- Input handling ---

    def _prepare(self, seq1: str, seq2: str, full_matrix: bool = True) -> Tuple[str, str]:
        for s in (seq1, seq2):
            if not isinstance(s, str):
                raise TypeError(f"sequences must be str, got {type(s).__name__}")
            if not s.isascii():
                # The C++ core compares bytes; multi-byte UTF-8 would silently misalign.
                raise ValueError("sequences must be ASCII")
        if full_matrix and (len(seq1) + 1) * (len(seq2) + 1) > self.max_cells:
            raise ValueError(
                f"{len(seq1)}x{len(seq2)} needs a full DP matrix larger than max_cells={self.max_cells:,}; "
                "use hirschberg() (linear space) or raise max_cells"
            )
        if not self.case_sensitive:
            seq1, seq2 = seq1.upper(), seq2.upper()
        return seq1, seq2

    def _sub(self, a: str, b: str) -> int:
        return self.match_score if a == b else self.mismatch_penalty

    # --- Public API ---

    def needleman_wunsch(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Global alignment using standard linear gap penalty."""
        seq1, seq2 = self._prepare(seq1, seq2)
        if self.use_cpp:
            return aligner_core.needleman_wunsch_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._needleman_wunsch_py(seq1, seq2)

    def smith_waterman(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Local alignment for optimal motif/subsequence identification."""
        seq1, seq2 = self._prepare(seq1, seq2)
        if self.use_cpp:
            return aligner_core.smith_waterman_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._smith_waterman_py(seq1, seq2)

    def gotoh(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Global alignment with Affine Gap Penalties (W = open + k * extend)."""
        seq1, seq2 = self._prepare(seq1, seq2)
        if self.use_cpp:
            return aligner_core.gotoh_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_open, self.gap_extend
            )
        return self._gotoh_py(seq1, seq2)

    def hirschberg(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Global alignment (same score as Needleman-Wunsch) in O(min(m, n)) DP memory."""
        seq1, seq2 = self._prepare(seq1, seq2, full_matrix=False)
        if self.use_cpp:
            return aligner_core.hirschberg_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._hirschberg_py(seq1, seq2)

    def score(self, seq1: str, seq2: str) -> int:
        """Optimal global (Needleman-Wunsch) score only, in O(min(m, n)) memory, no traceback."""
        seq1, seq2 = self._prepare(seq1, seq2, full_matrix=False)
        if len(seq2) > len(seq1):
            seq1, seq2 = seq2, seq1
        if self.use_cpp:
            return aligner_core.nw_score_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._nw_last_row_py(seq1, seq2)[-1]

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
                match = dp[i - 1][j - 1] + self._sub(seq1[i - 1], seq2[j - 1])
                delete = dp[i - 1][j] + self.gap_penalty
                insert = dp[i][j - 1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)

        aligned1, aligned2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + self._sub(seq1[i - 1], seq2[j - 1]):
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
                match = dp[i - 1][j - 1] + self._sub(seq1[i - 1], seq2[j - 1])
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
            match = dp[i - 1][j - 1] + self._sub(seq1[i - 1], seq2[j - 1])
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

    def _gotoh_py(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Mirror of gotoh_cpp (same recurrences and tie-breaking)."""
        m, n = len(seq1), len(seq2)
        go, ge = self.gap_open, self.gap_extend
        M = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
        X = [[NEG_INF] * (n + 1) for _ in range(m + 1)]  # gap in seq2 (consumes seq1)
        Y = [[NEG_INF] * (n + 1) for _ in range(m + 1)]  # gap in seq1 (consumes seq2)
        M[0][0] = 0
        for i in range(1, m + 1):
            X[i][0] = go + i * ge
        for j in range(1, n + 1):
            Y[0][j] = go + j * ge

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + self._sub(seq1[i - 1], seq2[j - 1])
                X[i][j] = max(M[i - 1][j] + go + ge, X[i - 1][j] + ge)
                Y[i][j] = max(M[i][j - 1] + go + ge, Y[i][j - 1] + ge)

        best = max(M[m][n], X[m][n], Y[m][n])
        # Same end-state preference as the C++ core on ties: X, then Y, then M.
        state = "X" if best == X[m][n] else ("Y" if best == Y[m][n] else "M")
        aligned1, aligned2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if state == "M":
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                prev = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
                state = "M" if prev == M[i - 1][j - 1] else ("X" if prev == X[i - 1][j - 1] else "Y")
                i -= 1
                j -= 1
            elif state == "X":
                aligned1.append(seq1[i - 1])
                aligned2.append('-')
                state = "M" if X[i][j] == M[i - 1][j] + go + ge else "X"
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j - 1])
                state = "M" if Y[i][j] == M[i][j - 1] + go + ge else "Y"
                j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), best

    def _nw_last_row_py(self, seq1: str, seq2: str) -> list:
        """Last row of the NW matrix using two rows of length len(seq2) + 1."""
        g = self.gap_penalty
        prev = [j * g for j in range(len(seq2) + 1)]
        for i in range(1, len(seq1) + 1):
            curr = [i * g] + [0] * len(seq2)
            a = seq1[i - 1]
            for j in range(1, len(seq2) + 1):
                curr[j] = max(prev[j - 1] + self._sub(a, seq2[j - 1]), prev[j] + g, curr[j - 1] + g)
            prev = curr
        return prev

    def _hirschberg_rec(self, seq1: str, seq2: str) -> Tuple[str, str]:
        m, n = len(seq1), len(seq2)
        if m == 0:
            return "-" * n, seq2
        if n == 0:
            return seq1, "-" * m
        if m == 1 or n == 1:
            a1, a2, _ = self._needleman_wunsch_py(seq1, seq2)
            return a1, a2

        mid = m // 2
        left = self._nw_last_row_py(seq1[:mid], seq2)
        right = self._nw_last_row_py(seq1[mid:][::-1], seq2[::-1])
        split = max(range(n + 1), key=lambda j: (left[j] + right[n - j], -j))  # first max, like C++

        l1, l2 = self._hirschberg_rec(seq1[:mid], seq2[:split])
        r1, r2 = self._hirschberg_rec(seq1[mid:], seq2[split:])
        return l1 + r1, l2 + r2

    def _hirschberg_py(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        swap = len(seq2) > len(seq1)  # keep the DP rows over the shorter sequence
        a1, a2 = self._hirschberg_rec(seq2, seq1) if swap else self._hirschberg_rec(seq1, seq2)
        if swap:
            a1, a2 = a2, a1
        score = sum(
            self.gap_penalty if c1 == "-" or c2 == "-" else self._sub(c1, c2) for c1, c2 in zip(a1, a2)
        )
        return a1, a2, score
