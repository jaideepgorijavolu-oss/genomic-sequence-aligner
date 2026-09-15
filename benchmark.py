import time
import random
import matplotlib.pyplot as plt
from aligner import SequenceAligner

try:
    from Bio.Align import PairwiseAligner
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


def generate_random_sequence(length: int) -> str:
    return "".join(random.choices(["A", "C", "G", "T"], k=length))


def time_custom_aligner(aligner_instance, s1: str, s2: str, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        aligner_instance.needleman_wunsch(s1, s2)
        times.append(time.perf_counter() - start)
    return min(times)


def time_biopython_aligner(bio_aligner, s1: str, s2: str, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        # Find one optimal alignment path rather than enumerating all co-optimal paths
        alignments = bio_aligner.align(s1, s2)
        _ = alignments[0]
        times.append(time.perf_counter() - start)
    return min(times)


def run_benchmarks():
    py_aligner = SequenceAligner(use_cpp=False)
    cpp_aligner = SequenceAligner(use_cpp=True)

    bio_aligner = None
    if BIOPYTHON_AVAILABLE:
        bio_aligner = PairwiseAligner(mode="global")
        bio_aligner.match_score = 2
        bio_aligner.mismatch_score = -1
        bio_aligner.open_gap_score = -2
        bio_aligner.extend_gap_score = -2

    lengths = [50, 100, 200, 400, 800, 1200]
    py_times = []
    cpp_times = []
    bio_times = []

    header = f"{'Length':<8} | {'Pure Py (s)':<12} | {'C++ (s)':<10} | {'Biopython (s)':<14} | {'vs Py':<8} | {'vs Bio':<8}"
    print(header)
    print("-" * len(header))

    for length in lengths:
        s1 = generate_random_sequence(length)
        s2 = generate_random_sequence(length)

        # 1. Pure Python
        t_py = time_custom_aligner(py_aligner, s1, s2)
        py_times.append(t_py)

        # 2. C++ (pybind11)
        t_cpp = time_custom_aligner(cpp_aligner, s1, s2)
        cpp_times.append(t_cpp)

        # 3. Biopython (C backend)
        t_bio = 0.0
        if BIOPYTHON_AVAILABLE:
            t_bio = time_biopython_aligner(bio_aligner, s1, s2)
            bio_times.append(t_bio)

        speedup_vs_py = f"{t_py / t_cpp:.1f}x" if t_cpp > 0 else "N/A"
        speedup_vs_bio = f"{t_bio / t_cpp:.1f}x" if (t_cpp > 0 and t_bio > 0) else "N/A"

        print(
            f"{length:<8} | {t_py:<12.5f} | {t_cpp:<10.5f} | "
            f"{t_bio:<14.5f} | {speedup_vs_py:<8} | {speedup_vs_bio:<8}"
        )

    # Visualization
    plt.figure(figsize=(9, 5))
    plt.plot(lengths, py_times, label="Pure Python", marker="o", color="#d9534f")
    if BIOPYTHON_AVAILABLE:
        plt.plot(lengths, bio_times, label="Biopython (Bio.Align)", marker="^", color="#f0ad4e")
    plt.plot(lengths, cpp_times, label="Custom C++ (pybind11)", marker="s", color="#0275d8")

    plt.title("Pairwise Global Alignment Runtime: Python vs Biopython vs Custom C++")
    plt.xlabel("Sequence Length (bp)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_benchmark.png", dpi=300)
    print("\nBenchmark complete. Saved updated scaling curve to complexity_benchmark.png")


if __name__ == "__main__":
    run_benchmarks()