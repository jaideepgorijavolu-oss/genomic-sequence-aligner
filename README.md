# Pairwise Genomic Sequence Alignment Engine (C++ / Python)

A high-performance algorithmic engine implementing dynamic programming algorithms for pairwise biological sequence alignment (DNA/RNA/Protein): **Needleman-Wunsch** (Global Alignment) and **Smith-Waterman** (Local Alignment).

Features a modular dual-backend architecture: an optimized **C++ extension via `pybind11`** utilizing contiguous 1D row-major indexing for cache efficiency, alongside an interpretable pure-Python reference implementation with automatic fallback.

---

## Algorithms Implemented

### 1. Needleman-Wunsch (Global Alignment)
- **Goal:** Finds optimal end-to-end alignment between two sequences across their entire lengths.
- **Recurrence Relation:**
  $$M[i, j] = \max \begin{cases} M[i-1, j-1] + S(x_i, y_j) \\ M[i-1, j] + d \\ M[i, j-1] + d \end{cases}$$
- **Complexity:** $\mathcal{O}(m \cdot n)$ time, $\mathcal{O}(m \cdot n)$ space.

### 2. Smith-Waterman (Local Alignment)
- **Goal:** Identifies the highest-scoring local homologous subregion/motif embedded within divergent or noisy sequences.
- **Recurrence Relation:**
  $$M[i, j] = \max \begin{cases} 0 \\ M[i-1, j-1] + S(x_i, y_j) \\ M[i-1, j] + d \\ M[i, j-1] + d \end{cases}$$
- **Traceback:** Initiates at the maximum score across the matrix and terminates upon encountering a cell with score 0.

---

## Features

- **Dual-Backend Architecture:** Seamlessly toggle between compiled C++ and pure Python backends via `use_cpp=True/False`.
- **Hardware & Memory Optimization:** C++ core utilizes flattened 1D `std::vector<int>` memory indexing to guarantee contiguous memory layout, maximizing L1/L2 cache locality and bypassing CPython object-boxing overhead.
- **Up to 80.5x Execution Speedup:** Reduces alignment latency on 1,200 bp sequences from ~0.69s down to 8.9ms.
- **Pointer Traceback:** Complete path reconstruction outputting aligned sequences with gap characters (`-`).
- **Empirical Benchmarking & Testing:** Automated runtime profiling across sequence scaling tiers ($50 \times 50$ to $1200 \times 1200$) with Pytest verification across both backends.

---

## Performance Benchmark

Benchmarked across matching sequence lengths ($50$ to $1,200\text{ bp}$) comparing pure Python against the C++ extension (`benchmark.py`):

| Sequence Length (bp) | Pure Python (s) | C++ Extension (s) | Speedup |
|:---------------------|:----------------|:------------------|:--------|
| 50                   | 0.00109         | 0.00003           | 38.4x   |
| 100                  | 0.00388         | 0.00006           | 67.8x   |
| 200                  | 0.01588         | 0.00022           | 72.9x   |
| 400                  | 0.06810         | 0.00085           | 80.5x   |
| 800                  | 0.28949         | 0.00423           | 68.4x   |
| 1200                 | 0.68748         | 0.00892           | 77.1x   |

![Benchmark Plot](complexity_benchmark.png)

---

## Project Structure

```text
genomic-sequence-aligner/
├── src/
│   └── aligner_core.cpp      # C++ dynamic programming inner loops & pybind11 module
├── aligner.py                # Python interface with dynamic C++/Python backend selection
├── benchmark.py              # Empirical runtime profiler & matplotlib plotting
├── test_aligner.py           # Pytest unit verification suite
├── setup.py                  # C++ pybind11 extension compilation script
├── complexity_benchmark.png  # Generated runtime scaling comparison
├── requirements.txt          # Python dependencies
└── README.md