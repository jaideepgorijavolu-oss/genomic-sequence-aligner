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
- **Competitive Throughput:** Consistently achieves up to a **93.9x speedup over pure Python** and executes **1.5x–1.9x faster than Biopython's C core**.
- **Pointer Traceback:** Complete path reconstruction outputting aligned sequences with gap characters (`-`).
- **Empirical Benchmarking & Testing:** Automated runtime profiling across sequence scaling tiers ($50 \times 50$ to $1200 \times 1200$) comparing Pure Python, Biopython (`Bio.Align`), and the custom C++ engine with automated Pytest verification.

---

## Performance Benchmark

Benchmarked across identical random DNA sequences comparing pure Python, Biopython (`Bio.Align.PairwiseAligner`), and the custom C++ core (`benchmark.py`):

| Sequence Length (bp) | Pure Python (s) | Biopython (s) | Custom C++ (s) | Speedup vs Py | Speedup vs Bio |
|:---------------------|:----------------|:--------------|:---------------|:--------------|:---------------|
| 50                   | 0.00109         | 0.00002       | 0.00002        | 65.0x         | 1.5x           |
| 100                  | 0.00542         | 0.00011       | 0.00006        | 93.9x         | 1.8x           |
| 200                  | 0.01856         | 0.00043       | 0.00022        | 84.8x         | 1.9x           |
| 400                  | 0.06480         | 0.00166       | 0.00085        | 76.2x         | 1.9x           |
| 800                  | 0.28018         | 0.00665       | 0.00407        | 68.8x         | 1.6x           |
| 1200                 | 0.68223         | 0.01518       | 0.00872        | 78.3x         | 1.7x           |

![Benchmark Plot](complexity_benchmark.png)

---

## Project Structure

```text
genomic-sequence-aligner/
├── src/
│   └── aligner_core.cpp      # C++ dynamic programming inner loops & pybind11 module
├── aligner.py                # Python interface with dynamic C++/Python backend selection
├── benchmark.py              # 3-way empirical runtime profiler & matplotlib plotting
├── test_aligner.py           # Pytest unit verification suite
├── setup.py                  # C++ pybind11 extension compilation script
├── complexity_benchmark.png  # Generated runtime scaling comparison
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Installation & Environment Setup

```bash
git clone https://github.com/jaideepgorijavolu-oss/genomic-sequence-aligner.git
cd genomic-sequence-aligner

python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Compile C++ Core Extension

Compile the optimized backend module in-place:

```bash
python setup.py build_ext --inplace
```

---

## Usage

```python
from aligner import SequenceAligner

# Initialize aligner (defaults to C++ backend if compiled)
aligner = SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2, use_cpp=True)

# 1. Global Alignment (Needleman-Wunsch)
seq1 = "GATTACA"
seq2 = "GCATGCU"
aligned1, aligned2, score = aligner.needleman_wunsch(seq1, seq2)
print(f"Global Alignment (Score: {score}):\n{aligned1}\n{aligned2}")

# 2. Local Alignment (Smith-Waterman)
seqA = "AAAAAGATTACATTTTT"
seqB = "CCCCCGATTACAGGGGG"
sub1, sub2, local_score = aligner.smith_waterman(seqA, seqB)
print(f"\nLocal Motif Match (Score: {local_score}):\n{sub1}\n{sub2}")
```

---

## Verification & Profiling

```bash
# Run test suite
pytest -v

# Run comparative 3-way benchmark and regenerate complexity plot
python benchmark.py
```