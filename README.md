# Pairwise Genomic Sequence Alignment Engine (C++ / Python)

**Highlights:** SSE2-vectorized striped Smith-Waterman (Farrar) with affine gaps and BLOSUM62 · multithreaded protein database search at **50 GCUPS (375× Biopython)** · Needleman-Wunsch, Gotoh and linear-space Hirschberg with full traceback · every result verified against Biopython.

A high-performance algorithmic engine implementing dynamic programming algorithms for pairwise biological sequence alignment (DNA/RNA/Protein): **Needleman-Wunsch** (Global Alignment), **Smith-Waterman** (Local Alignment), **Gotoh** (Affine Gap Penalties), and **Hirschberg** (Linear-Space Global Alignment).

Features a modular dual-backend architecture: an optimized **C++ extension via `pybind11`** utilizing contiguous 1D row-major indexing for cache efficiency, alongside an interpretable pure-Python reference implementation with automatic fallback.

---

## Algorithms Implemented

### 1. Needleman-Wunsch (Global Alignment)
- **Goal:** Optimal end-to-end alignment between sequences across full lengths.
- **Recurrence:** $M[i, j] = \max(M[i-1, j-1] + S(x_i, y_j), M[i-1, j] + d, M[i, j-1] + d)$
- **Complexity:** $\mathcal{O}(m \cdot n)$ time, $\mathcal{O}(m \cdot n)$ space.

### 2. Smith-Waterman (Local Alignment)
- **Goal:** Discovers highest-scoring local homologous motifs embedded within noisy sequences.
- **Recurrence:** $M[i, j] = \max(0, M[i-1, j-1] + S(x_i, y_j), M[i-1, j] + d, M[i, j-1] + d)$
- **Traceback:** Initiates at the global matrix maximum and terminates at cell score 0.

### 3. Gotoh Algorithm (Affine Gap Penalties)
- **Goal:** Biological realism where opening a gap costs more than extending it ($W = \text{open} + k \times \text{extend}$).
- **State Recurrence:** Employs a 3-state DP system tracking match ($M$), insertion ($I_x$), and deletion ($I_y$).
- **Complexity:** $\mathcal{O}(m \cdot n)$ time, $\mathcal{O}(m \cdot n)$ space.

### 4. Hirschberg's Algorithm (Linear-Space Alignment)
- **Goal:** Mitigates quadratic space exhaustion on chromosome-scale sequences.
- **Recurrence:** Combines forward and backward dynamic programming passes with divide-and-conquer recursion to compute optimal alignment tracebacks.
- **Complexity:** $\mathcal{O}(m \cdot n)$ time, $\mathcal{O}(\min(m, n))$ DP memory (shorter sequence is kept on the DP row).

### 5. Striped SIMD Smith-Waterman Search (Farrar 2007)
- **Goal:** Database search: score one query against thousands of sequences (the SSEARCH/SSW workload).
- **Scoring:** Affine gaps and any 256×256 substitution table: BLOSUM62 for proteins, match/mismatch for DNA.
- **Vectorization:** The query is interleaved into 8 stripes across the int16 lanes of an SSE2 register via a precomputed query profile; vertical-gap dependencies are resolved with Farrar's lazy-F loop. 16-bit saturating arithmetic, with automatic exact 32-bit rescoring when a score nears the int16 limit.
- **Parallelism:** `std::thread` pool with dynamic chunk scheduling across targets; the GIL is released for the whole search.
- **Portability:** SSE2 is baseline on x86-64 (MSVC, GCC, Clang); other CPUs (e.g. Apple Silicon) automatically use the scalar kernel.

---

## Features

- **Dual-Backend Architecture:** Toggle compiled C++ vs. pure Python backends via `use_cpp=True/False`.
- **Cache-Aligned Memory Optimization:** C++ core utilizes flattened 1D contiguous vectors (`std::vector<int>`) to maximize L1/L2 cache locality and bypass CPython boxing.
- **Verified Throughput:** See the benchmark table; speedups are only reported for runs whose scores match Biopython.
- **GIL-free kernels:** C++ calls release the GIL, so alignments in multiple threads run in parallel.
- **CLI:** `seqalign` / `python align_cli.py` aligns raw sequences or FASTA files with a BLAST-style view and identity stats.
- **Safe inputs:** case-insensitive by default, ASCII-only, and a `max_cells` guard that points to Hirschberg instead of exhausting RAM.
- **Full Traceback Reconstruction:** Outputs aligned query/reference sequence pairs formatted with gap tokens (`-`).
- **Database Search:** `LocalSearch(query, matrix="BLOSUM62").top(database, k=10)` and `seqalign search query.fa db.fa`.
- **Comprehensive Verification:** 48 pytest cases: every algorithm on both backends, randomized optimality checks against Biopython (including BLOSUM62 local search across four gap settings), SIMD-vs-scalar equivalence on odd query lengths and the int16 overflow path, deterministic multithreading, alignment validity (gap-stripped strings reproduce the inputs, rescored alignment equals the reported score), exact C++/Python parity, edge cases; CI on Linux/Windows/macOS.

---

## Database Search Benchmark

`python benchmark_search.py`: a 350-residue query against 5,000 synthetic proteins (1.72M residues, UniProt amino-acid composition, lognormal lengths, planted homologs), BLOSUM62 with BLASTP gap costs (open 11, extend 1). GCUPS = query length × database residues / seconds. Scores from every engine are asserted identical to Biopython's before timing. Measured on an AMD Ryzen 9 270 (8 cores / 16 threads), Windows, clang -O3 build.

| Engine | Threads | GCUPS | Speedup vs Biopython |
|:---|---:|---:|---:|
| Biopython PairwiseAligner (C) | 1 | 0.134 | 1.0x |
| C++ scalar (int32) | 1 | 0.208 | 1.5x |
| C++ SSE2 striped | 1 | 5.674 | 42.3x |
| C++ SSE2 striped | 2 | 10.646 | 79.3x |
| C++ SSE2 striped | 4 | 20.124 | 149.9x |
| C++ SSE2 striped | 8 | 35.016 | 260.8x |
| C++ SSE2 striped | 16 | 50.351 | 375.1x |

The vectorized kernel is 27× faster than the scalar kernel on one core, and thread scaling is near-linear up to the 8 physical cores, with a further 1.4× from SMT.

![Search scaling](search_scaling.png)

## Pairwise Alignment Benchmark

`python benchmark.py`: random DNA pairs (5 per length, median of 5 runs each); every configuration is checked to return Biopython's optimal score before it is timed. "Full" = alignment with traceback, "score-only" = linear-space DP without traceback (`SequenceAligner.score` vs `PairwiseAligner.score`). Measured on Windows x64, Python 3.14, clang -O3 build; results vary by machine and compiler.

| Length (bp) | Python full (s) | C++ full (s) | Biopython full (s) | C++ score-only (s) | Biopython score-only (s) | C++ vs Py | C++ vs Bio (full) | C++ vs Bio (score) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.00298 | 0.00002 | 0.00005 | 0.00001 | 0.00002 | 151.8x | 2.6x | 1.6x |
| 250 | 0.01836 | 0.00011 | 0.00053 | 0.00008 | 0.00013 | 168.7x | 4.8x | 1.7x |
| 500 | 0.08134 | 0.00061 | 0.00191 | 0.00030 | 0.00055 | 132.4x | 3.1x | 1.8x |
| 1000 | 0.33946 | 0.00233 | 0.00784 | 0.00123 | 0.00214 | 145.5x | 3.4x | 1.7x |
| 2000 | - | 0.00963 | 0.03164 | 0.00521 | 0.00891 | - | 3.3x | 1.7x |
| 4000 | - | 0.03839 | 0.12345 | 0.01980 | 0.03516 | - | 3.2x | 1.8x |

![Benchmark Plot](complexity_benchmark.png)

---

## Project Structure

```text
genomic-sequence-aligner/
├── src/
│   └── aligner_core.cpp      # C++ kernels: NW, SW, Gotoh, Hirschberg, SSE2 striped search
├── aligner.py                # Python API exposing dynamic C++/Python backend selection
├── benchmark.py              # 3-way empirical runtime profiler & matplotlib plotting
├── benchmark_search.py       # GCUPS / thread-scaling benchmark for SIMD database search
├── align_cli.py              # seqalign CLI: pairwise alignment and database search
├── test_search.py            # LocalSearch verification (Biopython, SIMD vs scalar, threads)
├── test_aligner.py           # Pytest unit verification suite
├── setup.py                  # C++ extension compilation script
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

```bash
python setup.py build_ext --inplace   # or: pip install .  (also installs the `seqalign` CLI)
```

### 3. Command Line

```bash
python align_cli.py ACGTTTTACG ACGACG --mode affine
python align_cli.py query.fasta reference.fasta --mode local
python align_cli.py search query.fasta database.fasta --matrix BLOSUM62 --top 10   # ranked hits
```

---

## Usage

```python
from aligner import SequenceAligner

# Initialize aligner (defaults to C++ backend if compiled)
aligner = SequenceAligner(
    match_score=2, 
    mismatch_penalty=-1, 
    gap_penalty=-2, 
    gap_open=-3, 
    gap_extend=-1, 
    use_cpp=True
)

# 1. Global Alignment (Needleman-Wunsch)
a1, a2, score = aligner.needleman_wunsch("GATTACA", "GCATGCU")
print(f"Global Alignment (Score: {score}):\n{a1}\n{a2}")

# 2. Local Alignment (Smith-Waterman)
sub1, sub2, loc_score = aligner.smith_waterman("AAAAAGATTACATTTTT", "CCCCCGATTACAGGGGG")
print(f"\nLocal Motif (Score: {loc_score}):\n{sub1}\n{sub2}")

# 3. Affine Gap Penalty Alignment (Gotoh)
g1, g2, gotoh_score = aligner.gotoh("ACGTACGT", "ACGTCGT")
print(f"\nGotoh Affine (Score: {gotoh_score}):\n{g1}\n{g2}")

# 4. Linear-Space Global Alignment (Hirschberg)
h1, h2, hirsch_score = aligner.hirschberg("ACGTGACTGATCG", "ACTTGATCGATC")
print(f"\nHirschberg Linear-Space (Score: {hirsch_score}):\n{h1}\n{h2}")
```

---

### Database search

```python
from aligner import LocalSearch

search = LocalSearch("MKTAYIAKQRQISFVKSHFSRQ", matrix="BLOSUM62", gap_open=-11, gap_extend=-1)
search.score("MKTAYIAKQRQLSFVKSHF")          # best local alignment score
search.search(database_sequences, threads=0)  # scores for every target, all cores
search.top(database_sequences, k=10)          # [(index, score), ...] best first
```

## Verification & Profiling

```bash
# Run unit test suite across all 4 algorithms
pytest -v

# Run comparative 3-way benchmark against Biopython
python benchmark.py

# Database-search throughput (GCUPS) and thread scaling
python benchmark_search.py
```