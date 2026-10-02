"""
Runtime benchmark: pure Python vs custom C++ vs Biopython (C), for
  * full global alignment with traceback (Needleman-Wunsch), and
  * score-only global alignment (linear-space DP, no traceback).

Every timed configuration is first checked to return the same optimal score as
Biopython, so speedups are never reported for a wrong answer.
"""
import argparse
import csv
import random
import statistics
import time

from aligner import CPP_AVAILABLE, SequenceAligner

try:
    from Bio.Align import PairwiseAligner
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

MATCH, MISMATCH, GAP = 2, -1, -2


def random_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choices("ACGT", k=length))


def median_time(fn, s1: str, s2: str, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(s1, s2)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def bio_full(bio):
    # One optimal alignment path (Biopython computes the trace matrix, then walks one path).
    return lambda a, b: bio.align(a, b)[0]


def run(args):
    if not CPP_AVAILABLE:
        raise SystemExit("C++ extension not built: python setup.py build_ext --inplace")
    rng = random.Random(args.seed)
    py = SequenceAligner(MATCH, MISMATCH, GAP, use_cpp=False)
    cpp = SequenceAligner(MATCH, MISMATCH, GAP, use_cpp=True, max_cells=10**9)

    bio = None
    if BIOPYTHON_AVAILABLE:
        bio = PairwiseAligner(mode="global")
        bio.match_score, bio.mismatch_score = MATCH, MISMATCH
        bio.open_gap_score = bio.extend_gap_score = GAP

    rows = []
    for length in args.lengths:
        pairs = [(random_dna(rng, length), random_dna(rng, length)) for _ in range(args.pairs)]
        if bio is not None:
            for s1, s2 in pairs:
                expected = bio.score(s1, s2)
                assert cpp.needleman_wunsch(s1, s2)[2] == expected == cpp.score(s1, s2), "score mismatch"

        def per_pair(fn):
            return statistics.median(median_time(fn, s1, s2, args.repeats) for s1, s2 in pairs)

        row = {"length": length,
               "cpp_full": per_pair(cpp.needleman_wunsch),
               "cpp_score": per_pair(cpp.score),
               "python_full": per_pair(py.needleman_wunsch) if length <= args.max_python_len else None,
               "bio_full": per_pair(bio_full(bio)) if bio else None,
               "bio_score": per_pair(bio.score) if bio else None}
        rows.append(row)
        print(fmt_row(row), flush=True)
    return rows


def ratio(a, b):
    return f"{a / b:.1f}x" if a and b else "-"


def secs(t):
    return f"{t:.5f}" if t is not None else "-"


HEADER = ("| Length (bp) | Python full (s) | C++ full (s) | Biopython full (s) | C++ score-only (s) "
          "| Biopython score-only (s) | C++ vs Py | C++ vs Bio (full) | C++ vs Bio (score) |\n"
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def fmt_row(r):
    return (f"| {r['length']} | {secs(r['python_full'])} | {secs(r['cpp_full'])} | {secs(r['bio_full'])} "
            f"| {secs(r['cpp_score'])} | {secs(r['bio_score'])} | {ratio(r['python_full'], r['cpp_full'])} "
            f"| {ratio(r['bio_full'], r['cpp_full'])} | {ratio(r['bio_score'], r['cpp_score'])} |")


def plot(rows, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    series = [("python_full", "Pure Python (full)", "o"), ("bio_full", "Biopython (full)", "^"),
              ("cpp_full", "C++ (full)", "s"), ("bio_score", "Biopython (score-only)", "v"),
              ("cpp_score", "C++ (score-only)", "D")]
    for key, label, marker in series:
        pts = [(r["length"], r[key]) for r in rows if r[key]]
        if pts:
            plt.plot(*zip(*pts), label=label, marker=marker)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Global alignment runtime (median over random DNA pairs)")
    plt.xlabel("Sequence length (bp)")
    plt.ylabel("Seconds (log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", type=int, nargs="+", default=[100, 250, 500, 1000, 2000, 4000])
    p.add_argument("--pairs", type=int, default=5, help="random sequence pairs per length")
    p.add_argument("--repeats", type=int, default=5, help="timed runs per pair (median)")
    p.add_argument("--max-python-len", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default="benchmark_results.csv")
    p.add_argument("--plot", default="complexity_benchmark.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    print(HEADER)
    rows = run(args)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plot:
        plot(rows, args.plot)
        print(f"\nSaved {args.csv} and {args.plot}")


if __name__ == "__main__":
    main()
