import time
import random
import matplotlib.pyplot as plt
from aligner import SequenceAligner

def generate_random_sequence(length: int) -> str:
    bases = ["A", "C", "G", "T"]
    return "".join(random.choices(bases, k=length))

def run_benchmarks():
    aligner = SequenceAligner()
    lengths = [50, 100, 200, 400, 800, 1200]
    nw_times = []
    sw_times = []

    print("Running sequence alignment empirical complexity benchmarks...\n")

    for n in lengths:
        s1 = generate_random_sequence(n)
        s2 = generate_random_sequence(n)

        # Benchmark Needleman-Wunsch
        start = time.perf_counter()
        aligner.needleman_wunsch(s1, s2)
        nw_duration = time.perf_counter() - start
        nw_times.append(nw_duration)

        # Benchmark Smith-Waterman
        start = time.perf_counter()
        aligner.smith_waterman(s1, s2)
        sw_duration = time.perf_counter() - start
        sw_times.append(sw_duration)

        print(f"Sequence Length: {n}x{n} | NW: {nw_duration:.4f}s | SW: {sw_duration:.4f}s")

    # Plot empirical O(n^2) scaling curve
    plt.figure(figsize=(8, 5))
    plt.plot(lengths, nw_times, marker="o", label="Needleman-Wunsch (Global)")
    plt.plot(lengths, sw_times, marker="s", label="Smith-Waterman (Local)")
    plt.title("Pairwise Alignment Runtime Scaling: Empirical O(n^2)")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_file = "complexity_benchmark.png"
    plt.savefig(output_file, dpi=300)
    print(f"\nComplexity plot saved as '{output_file}'.")

if __name__ == "__main__":
    run_benchmarks()