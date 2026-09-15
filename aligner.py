from typing import Tuple

try:
    import aligner_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


class SequenceAligner:
    """
    Pairwise Sequence Alignment Engine supporting:
    - Needleman-Wunsch (Global Alignment)
    - Smith-Waterman (Local Alignment)
    - Configurable C++ accelerated backend via pybind11
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_penalty: int = -1,
        gap_penalty: int = -2,
        use_cpp: bool = True
    ):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.use_cpp = use_cpp and CPP_AVAILABLE

    def _score(self, a: str, b: str) -> int:
        return self.match_score if a == b else self.mismatch_penalty

    def needleman_wunsch(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        if self.use_cpp:
            return aligner_core.needleman_wunsch_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._needleman_wunsch_py(seq1, seq2)

    def smith_waterman(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        if self.use_cpp:
            return aligner_core.smith_waterman_cpp(
                seq1, seq2, self.match_score, self.mismatch_penalty, self.gap_penalty
            )
        return self._smith_waterman_py(seq1, seq2)

    def _needleman_wunsch_py(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        m, n = len(seq1), len(seq2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            score_matrix[i][0] = i * self.gap_penalty
        for j in range(1, n + 1):
            score_matrix[0][j] = j * self.gap_penalty

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1][j - 1] + self._score(seq1[i - 1], seq2[j - 1])
                delete = score_matrix[i - 1][j] + self.gap_penalty
                insert = score_matrix[i][j - 1] + self.gap_penalty
                score_matrix[i][j] = max(match, delete, insert)

        aligned1, aligned2 = [], []
        i, j = m, n
        while i > 0 and j > 0:
            current_score = score_matrix[i][j]
            diagonal = score_matrix[i - 1][j - 1]
            up = score_matrix[i - 1][j]

            if current_score == diagonal + self._score(seq1[i - 1], seq2[j - 1]):
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif current_score == up + self.gap_penalty:
                aligned1.append(seq1[i - 1])
                aligned2.append("-")
                i -= 1
            else:
                aligned1.append("-")
                aligned2.append(seq2[j - 1])
                j -= 1

        while i > 0:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        while j > 0:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), score_matrix[m][n]

    def _smith_waterman_py(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        m, n = len(seq1), len(seq2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        max_score = 0
        max_pos = (0, 0)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1][j - 1] + self._score(seq1[i - 1], seq2[j - 1])
                delete = score_matrix[i - 1][j] + self.gap_penalty
                insert = score_matrix[i][j - 1] + self.gap_penalty
                score = max(0, match, delete, insert)
                score_matrix[i][j] = score

                if score > max_score:
                    max_score = score
                    max_pos = (i, j)

        aligned1, aligned2 = [], []
        i, j = max_pos
        while i > 0 and j > 0 and score_matrix[i][j] > 0:
            current_score = score_matrix[i][j]
            diagonal = score_matrix[i - 1][j - 1]
            up = score_matrix[i - 1][j]

            if current_score == diagonal + self._score(seq1[i - 1], seq2[j - 1]):
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif current_score == up + self.gap_penalty:
                aligned1.append(seq1[i - 1])
                aligned2.append("-")
                i -= 1
            else:
                aligned1.append("-")
                aligned2.append(seq2[j - 1])
                j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), max_score