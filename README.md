# Pairwise Genomic Sequence Alignment Engine

A lightweight algorithmic engine implementing dynamic programming algorithms for pairwise biological sequence alignment (DNA/RNA/Protein): **Needleman-Wunsch** (Global Alignment) and **Smith-Waterman** (Local Alignment).

Includes configurable scoring parameters (match rewards, mismatch penalties, linear gap penalties), pointer-based traceback path reconstruction, empirical complexity benchmarking, and a unit test suite.

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
- **Traceback:** Initiates at the maximum score in the entire matrix and terminates when encountering a cell with score 0.

---

## Features

- **Object-Oriented Design:** Configurable `SequenceAligner` class supporting custom match scores, mismatch penalties, and gap costs.
- **Pointer Traceback:** Reconstructs alignment strings with insertion/deletion gap characters (`-`).
- **Empirical Complexity Benchmarking:** Profiles execution time across variable input lengths ($50 \times 50$ to $1200 \times 1200$) to verify quadratic asymptotic scaling.
- **Automated Verification:** Comprehensive test suite testing exact matches, gap insertion logic, and conserved local motif isolation.

---

## Project Structure

```text
├── aligner.py                # Core dynamic programming alignment classes
├── benchmark.py              # Empirical runtime profiler & matplotlib plotting
├── test_aligner.py           # Pytest unit test suite
├── complexity_benchmark.png  # Generated runtime scaling curve
├── requirements.txt          # Project dependencies
└── README.md


Quickstart
1. Installation
git clone [https://github.com/jaideepgorijavolu-oss/genomic-sequence-aligner.git](https://github.com/jaideepgorijavolu-oss/genomic-sequence-aligner.git)
cd genomic-sequence-aligner

python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt