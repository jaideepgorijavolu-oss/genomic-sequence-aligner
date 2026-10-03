"""
Biopython agreement across many scoring regimes (not just the defaults), cross-algorithm
consistency, and input validation shared by every entry point.
"""
import random

import pytest

from aligner import CPP_AVAILABLE, LocalSearch, SequenceAligner

Align = pytest.importorskip("Bio.Align")

# (match, mismatch, linear gap, affine gap_open, affine gap_extend). Includes large mismatch
# penalties, where an insertion directly followed by a deletion beats a substitution, a
# zero-cost gap open, and a regime where opening is cheaper than extending.
REGIMES = [
    (2, -1, -2, -3, -1),
    (1, -10, -1, -1, -1),
    (5, -4, -3, -10, -1),
    (1, -1, -5, 0, -2),
    (3, -2, -1, -6, -2),
    (1, -3, -2, -1, -3),
    (2, -8, -4, -2, -1),
]

BACKENDS = ["python", "cpp"] if CPP_AVAILABLE else ["python"]


def pairs(n: int, seed: int, max_len: int = 14):
    rng = random.Random(seed)
    for _ in range(n):
        yield ("".join(rng.choices("ACGT", k=rng.randint(0, max_len))),
               "".join(rng.choices("ACGT", k=rng.randint(0, max_len))))


def bio(mode: str, match: int, mismatch: int, open_: int, extend: int):
    b = Align.PairwiseAligner(mode=mode)
    b.match_score, b.mismatch_score = match, mismatch
    b.open_gap_score, b.extend_gap_score = open_, extend
    return b


def rescore(a1, a2, match, mismatch, first_gap, extend):
    """Score an alignment from scratch: each gap run costs first_gap + (k - 1) * extend."""
    total, prev = 0, None
    for c1, c2 in zip(a1, a2):
        if c1 == "-" or c2 == "-":
            side = 1 if c1 == "-" else 2
            total += extend if prev == side else first_gap
            prev = side
        else:
            total += match if c1 == c2 else mismatch
            prev = None
    return total


def check_global(s1, s2, a1, a2):
    assert len(a1) == len(a2)
    assert a1.replace("-", "") == s1 and a2.replace("-", "") == s2
    assert not any(x == "-" and y == "-" for x, y in zip(a1, a2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("regime", REGIMES, ids=lambda r: "m{}_x{}_g{}_o{}_e{}".format(*r))
def test_all_algorithms_match_biopython_across_scoring(backend, regime):
    match, mismatch, gap, go, ge = regime
    a = SequenceAligner(match, mismatch, gap, go, ge, use_cpp=backend == "cpp")
    b_global = bio("global", match, mismatch, gap, gap)
    b_local = bio("local", match, mismatch, gap, gap)
    b_affine = bio("global", match, mismatch, go + ge, ge)

    for s1, s2 in pairs(80, seed=hash(regime) & 0xFFFF):
        x1, x2, nw = a.needleman_wunsch(s1, s2)
        check_global(s1, s2, x1, x2)
        assert rescore(x1, x2, match, mismatch, gap, gap) == nw
        h1, h2, hs = a.hirschberg(s1, s2)
        check_global(s1, s2, h1, h2)
        assert hs == nw == a.score(s1, s2)

        g1, g2, gs = a.gotoh(s1, s2)
        check_global(s1, s2, g1, g2)
        assert rescore(g1, g2, match, mismatch, go + ge, ge) == gs

        l1, l2, ls = a.smith_waterman(s1, s2)
        assert l1.replace("-", "") in s1 and l2.replace("-", "") in s2
        assert rescore(l1, l2, match, mismatch, gap, gap) == ls

        if s1 and s2:  # Biopython rejects empty sequences
            assert nw == b_global.score(s1, s2), (regime, s1, s2)
            assert gs == b_affine.score(s1, s2), (regime, s1, s2)
            assert ls == max(0, b_local.score(s1, s2)), (regime, s1, s2)


def test_reported_case_insertion_then_deletion():
    """mismatch -10, open -1, extend -1: 'A-'/'-T' (-4) beats the substitution (-10)."""
    for backend in BACKENDS:
        a = SequenceAligner(match_score=1, mismatch_penalty=-10, gap_open=-1, gap_extend=-1,
                            use_cpp=backend == "cpp")
        a1, a2, score = a.gotoh("A", "T")
        assert score == -4
        assert {(a1, a2)} <= {("A-", "-T"), ("-A", "T-")}


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")
@pytest.mark.parametrize("regime", REGIMES[:4], ids=lambda r: "m{}_x{}_g{}_o{}_e{}".format(*r))
def test_backends_identical_across_scoring(regime):
    py = SequenceAligner(*regime, use_cpp=False)
    cpp = SequenceAligner(*regime, use_cpp=True)
    for s1, s2 in pairs(100, seed=7, max_len=20):
        for name in ("needleman_wunsch", "smith_waterman", "gotoh", "hirschberg"):
            assert getattr(py, name)(s1, s2) == getattr(cpp, name)(s1, s2), (name, s1, s2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("match,mismatch,go,ge", [(2, -1, -11, -1), (1, -10, -1, -1), (3, -6, 0, -2), (1, -2, -5, -3)])
def test_local_search_matches_biopython_across_scoring(backend, match, mismatch, go, ge):
    b = bio("local", match, mismatch, go + ge, ge)
    for s1, s2 in pairs(60 if backend == "python" else 200, seed=11, max_len=30):
        if not s1 or not s2:
            continue
        s = LocalSearch(s1, match_score=match, mismatch_penalty=mismatch, gap_open=go, gap_extend=ge,
                        use_cpp=backend == "cpp")
        assert s.score(s2) == max(0, int(b.score(s1, s2))), (s1, s2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_four_global_algorithms_agree_on_valid_input(backend):
    """NW, Hirschberg, score-only DP, and Gotoh with a zero open cost are the same model."""
    rng = random.Random(3)
    a = SequenceAligner(2, -1, -2, gap_open=0, gap_extend=-2, use_cpp=backend == "cpp")
    for _ in range(100):
        s1 = "".join(rng.choices("ACGTN", k=rng.randint(0, 25)))
        s2 = "".join(rng.choices("ACGTN", k=rng.randint(0, 25)))
        nw = a.needleman_wunsch(s1, s2)[2]
        assert a.hirschberg(s1, s2)[2] == nw
        assert a.score(s1, s2) == nw
        assert a.gotoh(s1, s2)[2] == nw


BAD_INPUTS = ["A-C", "-", "AC GT", "ACG\nT", "AC\tG"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bad", BAD_INPUTS, ids=["dash", "only-dash", "space", "newline", "tab"])
def test_invalid_characters_raise_in_every_algorithm(backend, bad):
    a = SequenceAligner(use_cpp=backend == "cpp")
    for method in (a.needleman_wunsch, a.smith_waterman, a.gotoh, a.hirschberg, a.score):
        with pytest.raises(ValueError):
            method(bad, "ACGT")
        with pytest.raises(ValueError):
            method("ACGT", bad)
    with pytest.raises(ValueError):
        LocalSearch(bad, use_cpp=backend == "cpp")
    s = LocalSearch("ACGT", use_cpp=backend == "cpp")
    with pytest.raises(ValueError):
        s.score(bad)
    with pytest.raises(ValueError):
        s.search(["ACGT", bad])


def test_dash_input_is_rejected_by_the_cli(tmp_path, capsys):
    from align_cli import main

    for mode in ("global", "local", "affine", "linear-space"):
        with pytest.raises(SystemExit) as e:
            main(["A-C", "ACG", "--mode", mode])
        assert "'-'" in str(e.value)

    query = tmp_path / "q.fa"
    db = tmp_path / "db.fa"
    query.write_text(">q\nACGT\n")
    db.write_text(">aligned\nAC-GT\n")
    with pytest.raises(SystemExit) as e:
        main(["search", str(query), str(db), "--matrix", "none"])
    assert "'-'" in str(e.value)


def test_ambiguous_residues_are_ordinary_symbols():
    a = SequenceAligner(2, -1, -2, use_cpp=False)
    assert a.needleman_wunsch("ANA", "ANA")[2] == 6  # N matches N
    assert a.needleman_wunsch("ANA", "ACA")[2] == 3  # N vs C is a mismatch
