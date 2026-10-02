import os
import random

import pytest

from aligner import CPP_AVAILABLE, SequenceAligner, alignment_stats, format_alignment

MATCH, MISMATCH, GAP, GAP_OPEN, GAP_EXTEND = 2, -1, -2, -3, -1

# CI sets REQUIRE_CPP=1 so a failed build can't silently degrade to Python-only tests.
if os.environ.get("REQUIRE_CPP") == "1" and not CPP_AVAILABLE:
    raise RuntimeError("REQUIRE_CPP=1 but aligner_core is not importable; build the extension first")

BACKENDS = [
    "python",
    pytest.param("cpp", marks=pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")),
]


def make(backend: str, **kw) -> SequenceAligner:
    params = dict(match_score=MATCH, mismatch_penalty=MISMATCH, gap_penalty=GAP,
                  gap_open=GAP_OPEN, gap_extend=GAP_EXTEND, use_cpp=backend == "cpp")
    params.update(kw)
    a = SequenceAligner(**params)
    assert a.backend == backend
    return a


@pytest.fixture(params=BACKENDS)
def aligner(request):
    return make(request.param)


def random_pairs(n: int, max_len: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(n):
        yield (
            "".join(rng.choices("ACGT", k=rng.randint(0, max_len))),
            "".join(rng.choices("ACGT", k=rng.randint(0, max_len))),
        )


def rescore(a1: str, a2: str, affine: bool) -> int:
    """Score an alignment from scratch, independent of any DP matrix."""
    total, prev = 0, None
    for c1, c2 in zip(a1, a2):
        if c1 == "-" or c2 == "-":
            side = 1 if c1 == "-" else 2
            if affine:
                total += GAP_OPEN + GAP_EXTEND if prev != side else GAP_EXTEND
            else:
                total += GAP
            prev = side
        else:
            total += MATCH if c1 == c2 else MISMATCH
            prev = None
    return total


def assert_valid_global(s1, s2, a1, a2, score, affine=False):
    assert len(a1) == len(a2)
    assert a1.replace("-", "") == s1 and a2.replace("-", "") == s2
    assert not any(c1 == "-" and c2 == "-" for c1, c2 in zip(a1, a2))
    assert rescore(a1, a2, affine) == score


# --- Known answers ---------------------------------------------------------

def test_smith_waterman_finds_embedded_motif(aligner):
    a1, a2, score = aligner.smith_waterman("AAAGATTACATTT", "CCGATTACACC")
    assert (a1, a2, score) == ("GATTACA", "GATTACA", 14)


def test_gotoh_prefers_one_long_gap(aligner):
    a1, a2, score = aligner.gotoh("ACGTTTTACG", "ACGACG")
    assert (a1, a2) == ("ACGTTTTACG", "ACG----ACG")
    assert score == 6 * MATCH + GAP_OPEN + 4 * GAP_EXTEND


def test_identical_sequences(aligner):
    s = "GATTACA"
    for method in (aligner.needleman_wunsch, aligner.smith_waterman, aligner.gotoh, aligner.hirschberg):
        assert method(s, s) == (s, s, MATCH * len(s))


@pytest.mark.parametrize("s1,s2", [("", ""), ("", "ACG"), ("ACG", ""), ("A", "A"), ("A", "C"), ("A", "GATTACA")])
def test_edge_cases(aligner, s1, s2):
    for method, affine in ((aligner.needleman_wunsch, False), (aligner.hirschberg, False), (aligner.gotoh, True)):
        assert_valid_global(s1, s2, *method(s1, s2), affine=affine)
    a1, a2, score = aligner.smith_waterman(s1, s2)
    assert score >= 0 and len(a1) == len(a2)


# --- Properties on random inputs --------------------------------------------

def test_global_alignments_are_valid(aligner):
    for s1, s2 in random_pairs(300, 25):
        assert_valid_global(s1, s2, *aligner.needleman_wunsch(s1, s2))
        assert_valid_global(s1, s2, *aligner.hirschberg(s1, s2))
        assert_valid_global(s1, s2, *aligner.gotoh(s1, s2), affine=True)


def test_local_alignment_is_a_valid_substring_alignment(aligner):
    for s1, s2 in random_pairs(300, 25, seed=1):
        a1, a2, score = aligner.smith_waterman(s1, s2)
        assert a1.replace("-", "") in s1 and a2.replace("-", "") in s2
        assert rescore(a1, a2, affine=False) == score


def test_hirschberg_and_score_match_needleman_wunsch(aligner):
    pairs = list(random_pairs(200, 40, seed=2)) + [("ACGT" * 30, "AGT"), ("T", "ACGTACGTAC" * 8)]
    for s1, s2 in pairs:
        nw = aligner.needleman_wunsch(s1, s2)[2]
        assert aligner.hirschberg(s1, s2)[2] == nw
        assert aligner.score(s1, s2) == nw


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")
def test_cpp_and_python_backends_are_identical():
    py, cpp = make("python"), make("cpp")
    for s1, s2 in random_pairs(300, 30, seed=3):
        for name in ("needleman_wunsch", "smith_waterman", "gotoh", "hirschberg"):
            assert getattr(py, name)(s1, s2) == getattr(cpp, name)(s1, s2), (name, s1, s2)
        assert py.score(s1, s2) == cpp.score(s1, s2)


# --- Optimality against an independent implementation -------------------------

def test_scores_match_biopython(aligner):
    Align = pytest.importorskip("Bio.Align")

    def bio(mode, open_, extend):
        b = Align.PairwiseAligner(mode=mode)
        b.match_score, b.mismatch_score = MATCH, MISMATCH
        b.open_gap_score, b.extend_gap_score = open_, extend
        return b

    bio_global, bio_local = bio("global", GAP, GAP), bio("local", GAP, GAP)
    # Biopython's open score covers the first gap character; ours is open + extend.
    bio_affine = bio("global", GAP_OPEN + GAP_EXTEND, GAP_EXTEND)

    for s1, s2 in random_pairs(300, 30, seed=4):
        if not s1 or not s2:
            continue
        assert aligner.needleman_wunsch(s1, s2)[2] == bio_global.score(s1, s2)
        assert aligner.smith_waterman(s1, s2)[2] == max(0, bio_local.score(s1, s2))
        assert aligner.gotoh(s1, s2)[2] == bio_affine.score(s1, s2)


# --- Input handling -------------------------------------------------------------

def test_case_insensitive_by_default(aligner):
    assert aligner.needleman_wunsch("gattaca", "GATTACA")[2] == 7 * MATCH


def test_case_sensitive_option():
    a = SequenceAligner(use_cpp=False, case_sensitive=True)
    assert a.needleman_wunsch("a", "A")[2] == a.mismatch_penalty


def test_rejects_non_ascii_and_non_str():
    a = SequenceAligner(use_cpp=False)
    with pytest.raises(ValueError):
        a.needleman_wunsch("ACGÜ", "ACG")
    with pytest.raises(TypeError):
        a.needleman_wunsch(b"ACG", "ACG")
    with pytest.raises(TypeError):
        SequenceAligner(gap_penalty=-2.5)


def test_max_cells_guard_points_to_hirschberg():
    a = SequenceAligner(use_cpp=False, max_cells=100)
    with pytest.raises(ValueError, match="hirschberg"):
        a.needleman_wunsch("A" * 20, "A" * 20)
    assert a.hirschberg("A" * 20, "A" * 20)[2] == 20 * a.match_score  # linear space is exempt


# --- Utilities ---------------------------------------------------------------------

def test_alignment_stats_and_format():
    stats = alignment_stats("ACG--T", "AGGTTT")
    assert (stats.matches, stats.mismatches, stats.gaps, stats.gap_opens) == (3, 1, 2, 1)
    assert stats.identity == pytest.approx(0.5)
    assert format_alignment("ACG--T", "AGGTTT") == "ACG--T\n|.|  |\nAGGTTT"
