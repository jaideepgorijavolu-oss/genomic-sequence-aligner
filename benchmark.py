import time
import random
import matplotlib.pyplot as plt
from aligner import SequenceAligner

def generate_random_sequence(length: int) -> str:
    return "".join(random.choices(["A", "C", "G", "T"], k=length))

def time_alignment(aligner_instance, s1: str, s2: str, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        aligner_instance.needleman_wunsch(s1, s2)
        times.append(time.perf_counter() - start)
    return min(times)

def run_benchmarks():
    py_aligner = SequenceAligner(use_cpp=False)
    cpp_aligner = SequenceAligner(use_cpp=True)

    lengths = [50, 100, 200, 400, 800, 1200]
    py_times = []
    cpp_times = []

    print(f"{'Length':<10} | {'Python (s)':<12} | {'C++ (s)':<12} | {'Speedup':<10}")
    print("-" * 52)

    for l in lengths:
        s1 = generate_random_sequence(l)
        s2 = generate_random_sequence(l)

        # Time pure Python
        t_py = time_alignment(py_aligner, s1, s2)
        py_times.append(t_py)

        # Time C++ extension
        t_cpp = time_alignment(cpp_aligner, s1, s2)
        cpp_times.append(t_cpp)

        speedup = t_py / t_cpp if t_cpp > 0 else 0
        print(f"{l:<10} | {t_py:<12.5f} | {t_cpp:<12.5f} | {speedup:<10.1f}x")

    # Plot results
    plt.figure(figsize=(9, 5))
    plt.plot(lengths, py_times, label="Pure Python", marker="o", color="#d9534f")
    plt.plot(lengths, cpp_times, label="C++ (pybind11)", marker="s", color="#0275d8")
    plt.title("Needleman-Wunsch Execution Time vs Sequence Length")
    plt.xlabel("Sequence Length (bp)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_benchmark.png", dpi=300)
    print("\nSaved benchmark plot to complexity_benchmark.png")

if __name__ == "__main__":
    run_benchmarks()