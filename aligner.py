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


# --- Substitution matrices ------------------------------------------------------------

_BLOSUM62_ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX*"
_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""

MATRICES = {
    "BLOSUM62": {
        (a, b): int(v)
        for a, row in zip(_BLOSUM62_ALPHABET, _BLOSUM62_ROWS.split("\n")[1:-1])
        for b, v in zip(_BLOSUM62_ALPHABET, row.split())
    }
}


def substitution_table(matrix: str = None, match_score: int = 2, mismatch_penalty: int = -1,
                       case_sensitive: bool = False) -> list:
    """
    Flattened 256x256 score table indexed [a * 256 + b] by byte value. With a named matrix,
    letters are case-folded and symbols outside its alphabet score as 'X' (unknown residue).
    """
    fold = (lambda c: c) if case_sensitive and matrix is None else (lambda c: c.upper())
    chars = [fold(chr(x)) for x in range(256)]
    if matrix is None:
        return [match_score if a == b else mismatch_penalty for a in chars for b in chars]
    try:
        scores = MATRICES[matrix.upper()]
    except KeyError:
        raise ValueError(f"unknown matrix {matrix!r}; available: {sorted(MATRICES)}") from None
    alphabet = {a for a, _ in scores}
    chars = [c if c in alphabet else "X" for c in chars]
    return [scores[a, b] for a in chars for b in chars]


class LocalSearch:
    """
    One query against many targets: best local-alignment (Smith-Waterman) score with affine
    gaps (a gap of length k scores gap_open + k * gap_extend), using either a substitution
    matrix (e.g. "BLOSUM62") or match/mismatch scores.

    The C++ backend uses an SSE2 striped kernel (Farrar 2007) with multithreaded search;
    the Python backend is an exact reference implementation of the same recurrence.
    Defaults match BLASTP's gap costs (open 11, extend 1).
    """

    def __init__(self, query: str, matrix: str = None, match_score: int = 2, mismatch_penalty: int = -1,
                 gap_open: int = -11, gap_extend: int = -1, use_cpp: bool = True, case_sensitive: bool = False):
        _check_sequence(query)
        if gap_open > 0 or gap_extend > 0:
            raise ValueError("gap_open and gap_extend must be <= 0")
        self.query = query
        self.gap_open, self.gap_extend = gap_open, gap_extend
        self.table = substitution_table(matrix, match_score, mismatch_penalty, case_sensitive)
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self._core = aligner_core.LocalScorer(query, self.table, gap_open, gap_extend) if self.use_cpp else None

    @property
    def backend(self) -> str:
        if not self.use_cpp:
            return "python"
        return "cpp-sse2" if self._core.simd_enabled else "cpp-scalar"

    def score(self, target: str) -> int:
        _check_sequence(target)
        if self._core is not None:
            return self._core.score(target)
        return self._score_py(target)

    def search(self, targets, threads: int = 0) -> list:
        """Scores for every target, in order. threads=0 uses all cores (C++ backend)."""
        targets = list(targets)
        for t in targets:
            _check_sequence(t)
        if self._core is not None:
            return self._core.score_many(targets, threads)
        return [self._score_py(t) for t in targets]

    def top(self, targets, k: int = 10, threads: int = 0) -> list:
        """The k best (index, score) hits, highest score first."""
        scores = self.search(targets, threads)
        return sorted(enumerate(scores), key=lambda hit: (-hit[1], hit[0]))[:k]

    def _score_py(self, target: str) -> int:
        q = [ord(c) * 256 for c in self.query]
        m = len(q)
        if m == 0 or not target:
            return 0
        open_cost, ext_cost = -(self.gap_open + self.gap_extend), -self.gap_extend
        table = self.table
        H, E = [0] * (m + 1), [0] * (m + 1)
        best = 0
        for t in map(ord, target):
            diag = h_up = f = 0
            for i in range(1, m + 1):
                e = max(E[i] - ext_cost, H[i] - open_cost)
                f = max(f - ext_cost, h_up - open_cost)
                h = max(0, diag + table[q[i - 1] + t], e, f)
                diag, H[i], E[i], h_up = H[i], h, e, h
                if h > best:
                    best = h
        return best


def _check_sequence(s) -> None:
    """
    Accepted input: printable, non-whitespace ASCII except '-'. Every other character is an
    ordinary residue symbol: with match/mismatch scoring, identical symbols match (so 'N'
    matches 'N'); with a substitution matrix, symbols outside its alphabet score as 'X'.
    '-' is reserved as the gap character in alignment output, so a literal '-' in the input
    would be scored as a residue by the DP but as a gap when an alignment is rescored.
    Whitespace is rejected rather than silently stripped (the CLI strips FASTA line breaks).
    """
    if not isinstance(s, str):
        raise TypeError(f"sequences must be str, got {type(s).__name__}")
    if not s.isascii():
        raise ValueError("sequences must be ASCII")
    if "-" in s:
        raise ValueError("sequences must not contain '-' (reserved for gaps in alignment output); "
                         "remove gaps from pre-aligned input first")
    bad = next((c for c in s if not c.isprintable() or c.isspace()), None)
    if bad is not None:
        raise ValueError(f"sequences must not contain whitespace or control characters (found {bad!r})")


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
        # ASCII only: the C++ core compares bytes, so multi-byte UTF-8 would silently misalign.
        _check_sequence(seq1)
        _check_sequence(seq2)
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
                # A gap may directly follow a gap in the other sequence (a new gap opens).
                X[i][j] = max(M[i - 1][j] + go + ge, X[i - 1][j] + ge, Y[i - 1][j] + go + ge)
                Y[i][j] = max(M[i][j - 1] + go + ge, Y[i][j - 1] + ge, X[i][j - 1] + go + ge)

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
                here = X[i][j]
                state = "M" if here == M[i - 1][j] + go + ge else ("X" if here == X[i - 1][j] + ge else "Y")
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j - 1])
                here = Y[i][j]
                state = "M" if here == M[i][j - 1] + go + ge else ("Y" if here == Y[i][j - 1] + ge else "X")
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
