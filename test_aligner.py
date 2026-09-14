import pytest
from aligner import SequenceAligner

@pytest.fixture
def aligner():
    return SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2)

def test_needleman_wunsch_exact_match(aligner):
    seq = "GATTACA"
    a1, a2, score = aligner.needleman_wunsch(seq, seq)
    assert a1 == seq
    assert a2 == seq
    assert score == len(seq) * 2

def test_needleman_wunsch_alignment_with_gap(aligner):
    seq1 = "GATTACA"
    seq2 = "GCATGCU"
    a1, a2, score = aligner.needleman_wunsch(seq1, seq2)
    assert len(a1) == len(a2)
    assert a1.replace("-", "") == seq1
    assert a2.replace("-", "") == seq2

def test_smith_waterman_local_subsequence(aligner):
    # Conserved motif embedded in random flanking sequences
    seq1 = "AAAAAGATTACATTTTT"
    seq2 = "CCCCCGATTACAGGGGG"
    a1, a2, score = aligner.smith_waterman(seq1, seq2)
    
    assert a1 == "GATTACA"
    assert a2 == "GATTACA"
    assert score == 7 * 2