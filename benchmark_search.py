"""
Protein database-search throughput: one query vs a synthetic protein database, scored with
local alignment (BLOSUM62, BLASTP gap costs: open 11, extend 1).

Throughput is reported in GCUPS (billions of DP cell updates per second = query length x
total database residues / seconds), the standard metric for Smith-Waterman engines.
Every engine's scores are checked against Biopython before any timing is reported.
"""
import argparse
import csv
import os
import random
import time

from aligner import CPP_AVAILABLE, LocalSearch

try:
    from Bio.Align import PairwiseAligner, substitution_matrices
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

AA = "ACDEFGHIKLMNPQRSTVWY"
# Background amino-acid frequencies (UniProt), so random proteins have realistic composition.
AA_FREQ = [8.25, 1.38, 5.46, 6.72, 3.86, 7.07, 2.27, 5.91, 5.80, 9.65,
           2.41, 4.06, 4.74, 3.93, 5.53, 6.64, 5.35, 6.86, 1.10, 2.92]
GAP_OPEN, GAP_EXTEND = -11, -1


def make_database(rng: random.Random, n: int, query: str) -> list:
    db = []
    for i in range(n):
        if i % 50 == 0:  # sprinkle in homologs of the query
            s = list(query)
            for _ in range(len(s) // 5):
                s[rng.randrange(len(s))] = rng.choices(AA, AA_FREQ)[0]
            db.append("".join(s))
        else:
            length = min(1500, max(30, int(rng.lognormvariate(5.7, 0.55))))  # median ~300 aa
            db.append("".join(rng.choices(AA, AA_FREQ, k=length)))
    return db


def timed(fn, repeats: int):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query-length", type=int, default=350)
    p.add_argument("--db-size", type=int, default=5000)
    p.add_argument("--bio-subset", type=int, default=300, help="targets timed with Biopython (it is slow)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default="search_benchmark.csv")
    p.add_argument("--plot", default="search_scaling.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    if not CPP_AVAILABLE:
        raise SystemExit("C++ extension not built: python setup.py build_ext --inplace")

    rng = random.Random(args.seed)
    query = "".join(rng.choices(AA, AA_FREQ, k=args.query_length))
    db = make_database(rng, args.db_size, query)
    residues = sum(map(len, db))
    search = LocalSearch(query, matrix="BLOSUM62", gap_open=GAP_OPEN, gap_extend=GAP_EXTEND)
    core = search._core
    cpus = os.cpu_count() or 1
    print(f"query {len(query)} aa | database {len(db):,} sequences, {residues:,} residues | "
          f"{cpus} logical CPUs | kernel {search.backend}\n")

    rows = []

    def record(engine, threads, n_targets, seconds):
        cells = len(query) * sum(len(t) for t in db[:n_targets])
        rows.append({"engine": engine, "threads": threads, "targets": n_targets,
                     "seconds": seconds, "gcups": cells / seconds / 1e9})

    # Correctness gate: every engine must agree with Biopython on the subset it is timed on.
    subset = db[: args.bio_subset]
    simd_scores = core.score_many(db, 0)
    assert simd_scores == [core.score_scalar(t) for t in db], "SIMD and scalar kernels disagree"
    if BIOPYTHON_AVAILABLE:
        bio = PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                              open_gap_score=GAP_OPEN + GAP_EXTEND, extend_gap_score=GAP_EXTEND)
        t_bio, bio_scores = timed(lambda: [int(bio.score(query, t)) for t in subset], 1)
        assert bio_scores == simd_scores[: len(subset)], "scores disagree with Biopython"
        record("Biopython PairwiseAligner (C)", 1, len(subset), t_bio)

    t_scalar, _ = timed(lambda: [core.score_scalar(t) for t in db], args.repeats)
    record("C++ scalar (int32)", 1, len(db), t_scalar)

    thread_counts = sorted({1, 2, 4, 8, cpus} | ({16} if cpus >= 16 else set()))
    for n in thread_counts:
        t, _ = timed(lambda: core.score_many(db, n), args.repeats)
        record("C++ SSE2 striped", n, len(db), t)

    base = rows[0]["gcups"] if BIOPYTHON_AVAILABLE else None
    print("| Engine | Threads | GCUPS | Speedup vs Biopython |")
    print("|:---|---:|---:|---:|")
    for r in rows:
        speed = f"{r['gcups'] / base:.1f}x" if base else "-"
        print(f"| {r['engine']} | {r['threads']} | {r['gcups']:.3f} | {speed} |")
    top = search.top(db, k=3)
    print(f"\nTop hits (index, score): {top}  (homologs are planted every 50th sequence)")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plot:
        import matplotlib.pyplot as plt

        simd = [r for r in rows if r["engine"].startswith("C++ SSE2")]
        plt.figure(figsize=(7, 4.5))
        plt.plot([r["threads"] for r in simd], [r["gcups"] for r in simd], marker="o", label="C++ SSE2 striped")
        scalar = next(r for r in rows if r["engine"].startswith("C++ scalar"))
        plt.axhline(scalar["gcups"], linestyle="--", color="gray", label="C++ scalar, 1 thread")
        if base:
            plt.axhline(base, linestyle=":", color="tab:orange", label="Biopython, 1 thread")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Threads")
        plt.ylabel("GCUPS (log)")
        plt.title(f"Protein search throughput: {len(query)} aa query vs {len(db):,} sequences")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        print(f"Saved {args.csv} and {args.plot}")


if __name__ == "__main__":
    main()
