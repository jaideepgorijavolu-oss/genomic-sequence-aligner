import random

import pytest

from aligner import CPP_AVAILABLE, MATRICES, LocalSearch, SequenceAligner, substitution_table

AA = "ACDEFGHIKLMNPQRSTVWY"
GAP_SETTINGS = [(-11, -1), (-5, -2), (0, -4), (-1, -1)]  # includes linear gaps (open 0)

BACKENDS = [
    "python",
    pytest.param("cpp", marks=pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")),
]
requires_cpp = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")


def mutate(rng: random.Random, seq: str, edits: int) -> str:
    """A homologous sequence: random substitutions, insertions and deletions."""
    s = list(seq)
    for _ in range(edits):
        p = rng.randrange(len(s) + 1)
        op = rng.random()
        if op < 0.35:
            s.insert(p, rng.choice(AA))
        elif op < 0.7 and s:
            s.pop(min(p, len(s) - 1))
        elif s:
            s[min(p, len(s) - 1)] = rng.choice(AA)
    return "".join(s) or "A"


def protein_pairs(n: int, max_len: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n):
        q = "".join(rng.choices(AA, k=rng.randint(1, max_len)))
        t = mutate(rng, q, rng.randint(0, 8)) if rng.random() < 0.5 else "".join(
            rng.choices(AA, k=rng.randint(1, max_len)))
        yield q, t


def test_embedded_blosum62_matches_biopython():
    sm = pytest.importorskip("Bio.Align.substitution_matrices")
    bio = sm.load("BLOSUM62")
    ours = MATRICES["BLOSUM62"]
    for a, b in ours:
        assert ours[a, b] == int(bio[a][b]), (a, b)


@pytest.mark.parametrize("gap_open,gap_extend", GAP_SETTINGS)
def test_scores_match_biopython(gap_open, gap_extend, request):
    Align = pytest.importorskip("Bio.Align")
    bio = Align.PairwiseAligner(
        mode="local", substitution_matrix=Align.substitution_matrices.load("BLOSUM62"),
        open_gap_score=gap_open + gap_extend, extend_gap_score=gap_extend,
    )
    for backend in ("python", "cpp") if CPP_AVAILABLE else ("python",):
        n = 40 if backend == "python" else 400
        for q, t in protein_pairs(n, 60, seed=hash((gap_open, gap_extend)) & 0xFFFF):
            s = LocalSearch(q, matrix="BLOSUM62", gap_open=gap_open, gap_extend=gap_extend,
                            use_cpp=backend == "cpp")
            assert s.score(t) == max(0, int(bio.score(q, t))), (backend, q, t)


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_gaps_match_smith_waterman(backend):
    """With gap_open=0 the affine model reduces to the linear-gap Smith-Waterman kernel."""
    rng = random.Random(5)
    sw = SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2, use_cpp=backend == "cpp")
    for _ in range(100):
        q = "".join(rng.choices("ACGT", k=rng.randint(1, 40)))
        t = "".join(rng.choices("ACGT", k=rng.randint(1, 40)))
        s = LocalSearch(q, match_score=2, mismatch_penalty=-1, gap_open=0, gap_extend=-2, use_cpp=backend == "cpp")
        assert s.score(t) == sw.smith_waterman(q, t)[2]


@requires_cpp
def test_simd_matches_scalar_on_long_and_odd_lengths():
    rng = random.Random(9)
    for qlen in [1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 300, 1001]:
        q = "".join(rng.choices(AA, k=qlen))
        s = LocalSearch(q, matrix="BLOSUM62")
        assert s.backend == "cpp-sse2" or not s._core.simd_enabled
        for _ in range(10):
            t = mutate(rng, q, rng.randint(0, 40)) if rng.random() < 0.5 else "".join(
                rng.choices(AA, k=rng.randint(1, 1200)))
            assert s.score(t) == s._core.score_scalar(t), (qlen, len(t))


@requires_cpp
def test_int16_overflow_falls_back_to_exact_scores():
    q = "W" * 3200  # 3200 * 11 = 35200 > INT16_MAX
    assert LocalSearch(q, matrix="BLOSUM62").score(q) == 11 * 3200


@requires_cpp
def test_multithreaded_search_is_deterministic():
    rng = random.Random(3)
    q = "".join(rng.choices(AA, k=200))
    db = [mutate(rng, q, rng.randint(0, 60)) if i % 3 == 0 else "".join(rng.choices(AA, k=rng.randint(0, 400)))
          for i in range(500)]
    s = LocalSearch(q, matrix="BLOSUM62")
    serial = [s.score(t) for t in db]
    assert s.search(db, threads=1) == serial
    assert s.search(db, threads=8) == serial
    assert s.search(db) == serial


@pytest.mark.parametrize("backend", BACKENDS)
def test_top_hits_rank_homologs_first(backend):
    rng = random.Random(11)
    q = "".join(rng.choices(AA, k=80))
    decoys = ["".join(rng.choices(AA, k=80)) for _ in range(30)]
    db = decoys[:10] + [mutate(rng, q, 5)] + decoys[10:] + [q]
    hits = LocalSearch(q, matrix="BLOSUM62", use_cpp=backend == "cpp").top(db, k=2)
    assert [i for i, _ in hits] == [31, 10]
    assert hits[0][1] > hits[1][1] > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_edge_cases_and_symbol_handling(backend):
    cpp = backend == "cpp"
    assert LocalSearch("", matrix="BLOSUM62", use_cpp=cpp).score("ACD") == 0
    assert LocalSearch("ACD", matrix="BLOSUM62", use_cpp=cpp).search(["", "acd"]) == [0, 4 + 9 + 6]
    # Symbols outside the matrix alphabet score as 'X'.
    s = LocalSearch("AJA", matrix="BLOSUM62", use_cpp=cpp)
    assert s.score("AXA") == s.score("AJA")
    with pytest.raises(ValueError):
        LocalSearch("ACD", gap_open=1)
    with pytest.raises(ValueError):
        LocalSearch("ACD", matrix="PAM999")
    with pytest.raises(ValueError):
        LocalSearch("ACD", use_cpp=cpp).score("ACDÜ")


def test_substitution_table_case_handling():
    t = substitution_table(None, 2, -1, case_sensitive=False)
    assert t[ord("a") * 256 + ord("A")] == 2
    t = substitution_table(None, 2, -1, case_sensitive=True)
    assert t[ord("a") * 256 + ord("A")] == -1
