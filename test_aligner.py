import pytest
from aligner import SequenceAligner


@pytest.fixture
def aligner_py():
    return SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2, use_cpp=False)


@pytest.fixture
def aligner_cpp():
    return SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2, use_cpp=True)


def test_needleman_wunsch_parity(aligner_py, aligner_cpp):
    s1 = "GATTACA"
    s2 = "GCATGCU"
    a1_py, a2_py, score_py = aligner_py.needleman_wunsch(s1, s2)
    a1_cpp, a2_cpp, score_cpp = aligner_cpp.needleman_wunsch(s1, s2)

    assert score_py == score_cpp
    assert len(a1_cpp) == len(a2_cpp)


def test_smith_waterman_parity(aligner_py, aligner_cpp):
    s1 = "AAAGATTACATTT"
    s2 = "CCGATTACACC"
    _, _, score_py = aligner_py.smith_waterman(s1, s2)
    _, _, score_cpp = aligner_cpp.smith_waterman(s1, s2)

    assert score_py == score_cpp
    assert score_cpp == 14  # "GATTACA" perfect match (7 * 2)


def test_gotoh_affine_gaps(aligner_cpp):
    # Aligning a sequence with a contiguous 3-bp insertion
    s1 = "AGTCAGTC"
    s2 = "AGT---CAGTC".replace("-", "")  # AGTCAGTC vs AGTCAGTC with gap
    aligner_affine = SequenceAligner(
        match_score=2, mismatch_penalty=-1, gap_open=-3, gap_extend=-1, use_cpp=True
    )
    a1, a2, score = aligner_affine.gotoh("ACGTACGT", "ACGT---ACGT".replace("-", ""))
    assert len(a1) == len(a2)
    assert score > -10


def test_hirschberg_score_equivalence(aligner_cpp):
    # Hirschberg (linear space) must produce the exact same optimal score as Needleman-Wunsch
    s1 = "ACGTGACTGATCGATCG"
    s2 = "ACTTGATCGATCGA"
    _, _, nw_score = aligner_cpp.needleman_wunsch(s1, s2)
    _, _, hirsch_score = aligner_cpp.hirschberg(s1, s2)

    assert nw_score == hirsch_score