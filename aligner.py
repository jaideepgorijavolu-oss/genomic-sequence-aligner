from typing import Tuple, List

class SequenceAligner:
    """
    Pairwise Sequence Alignment Engine implementing:
    - Needleman-Wunsch (Global Alignment)
    - Smith-Waterman (Local Alignment)
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """

    def __init__(self, match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def _score(self, a: str, b: str) -> int:
        return self.match_score if a == b else self.mismatch_penalty

    def needleman_wunsch(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """
        Global pairwise sequence alignment using Dynamic Programming.
        Returns: (aligned_seq1, aligned_seq2, alignment_score)
        """
        m, n = len(seq1), len(seq2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases (linear gap penalty along borders)
        for i in range(1, m + 1):
            score_matrix[i][0] = i * self.gap_penalty
        for j in range(1, n + 1):
            score_matrix[0][j] = j * self.gap_penalty

        # Fill Dynamic Programming score matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1][j - 1] + self._score(seq1[i - 1], seq2[j - 1])
                delete = score_matrix[i - 1][j] + self.gap_penalty
                insert = score_matrix[i][j - 1] + self.gap_penalty
                score_matrix[i][j] = max(match, delete, insert)

        # Backtracking / Traceback to reconstruct optimal alignment
        aligned1, aligned2 = [], []
        i, j = m, n

        while i > 0 and j > 0:
            current_score = score_matrix[i][j]
            diagonal = score_matrix[i - 1][j - 1]
            up = score_matrix[i - 1][j]
            left = score_matrix[i][j - 1]

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

        final_score = score_matrix[m][n]
        return "".join(reversed(aligned1)), "".join(reversed(aligned2)), final_score

    def smith_waterman(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """
        Local pairwise sequence alignment using Dynamic Programming with 0-floor thresholding.
        Returns: (aligned_subseq1, aligned_subseq2, max_score)
        """
        m, n = len(seq1), len(seq2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]

        max_score = 0
        max_pos = (0, 0)

        # Fill Dynamic Programming matrix (floored at 0 for local regions)
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

        # Traceback from highest-scoring cell until reaching 0
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