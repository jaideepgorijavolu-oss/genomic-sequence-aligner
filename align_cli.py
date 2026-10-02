"""
Command-line pairwise aligner.

    seqalign GATTACA GCATGCU
    seqalign query.fasta reference.fasta --mode local
    python align_cli.py ACGTTTTACG ACGACG --mode affine --gap-open -5 --gap-extend -1

    seqalign search query.fasta database.fasta --matrix BLOSUM62 --top 10
"""
import argparse
import os
import sys

from aligner import LocalSearch, SequenceAligner, alignment_stats, format_alignment

MODES = {
    "global": "needleman_wunsch",
    "local": "smith_waterman",
    "affine": "gotoh",
    "linear-space": "hirschberg",
}


def read_fasta(path: str) -> list:
    """All (name, sequence) records in a FASTA file; a file without headers is one record."""
    records, name, chunks = [], os.path.basename(path), []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if chunks:
                    records.append((name, "".join(chunks)))
                name, chunks = (line[1:].split() or [os.path.basename(path)])[0], []
            elif line:
                chunks.append(line)
    if chunks:
        records.append((name, "".join(chunks)))
    return records


def read_sequence(arg: str) -> tuple:
    """Return (name, sequence) from a FASTA/plain-text file path (first record), or treat arg as a literal sequence."""
    if not os.path.isfile(arg):
        return "seq", arg.strip()
    records = read_fasta(arg)
    if not records:
        raise SystemExit(f"{arg}: no sequence found")
    return records[0]


def search_main(argv) -> int:
    p = argparse.ArgumentParser(prog="seqalign search",
                                description="Rank database sequences by local alignment score against a query")
    p.add_argument("query", help="query sequence or FASTA file (first record)")
    p.add_argument("database", help="FASTA file of target sequences")
    p.add_argument("--matrix", default="BLOSUM62", help="substitution matrix, or 'none' for match/mismatch")
    p.add_argument("--match", type=int, default=2)
    p.add_argument("--mismatch", type=int, default=-1)
    p.add_argument("--gap-open", type=int, default=-11)
    p.add_argument("--gap-extend", type=int, default=-1)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores")
    args = p.parse_args(argv)

    qname, query = read_sequence(args.query)
    records = read_fasta(args.database)
    if not records:
        raise SystemExit(f"{args.database}: no sequences found")
    matrix = None if args.matrix.lower() == "none" else args.matrix
    try:
        search = LocalSearch(query, matrix=matrix, match_score=args.match, mismatch_penalty=args.mismatch,
                             gap_open=args.gap_open, gap_extend=args.gap_extend)
        hits = search.top([seq for _, seq in records], k=args.top, threads=args.threads)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"# query {qname} ({len(query)} residues) vs {len(records):,} sequences | "
          f"matrix={matrix or 'match/mismatch'} gap_open={args.gap_open} gap_extend={args.gap_extend} "
          f"backend={search.backend}")
    print(f"{'rank':>4}  {'score':>6}  {'length':>6}  name")
    for rank, (i, score) in enumerate(hits, 1):
        name, seq = records[i]
        print(f"{rank:>4}  {score:>6}  {len(seq):>6}  {name}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "search":
        return search_main(argv[1:])
    p = argparse.ArgumentParser(description="Pairwise sequence alignment (C++ core with Python fallback)")
    p.add_argument("seq1", help="sequence or FASTA file")
    p.add_argument("seq2", help="sequence or FASTA file")
    p.add_argument("--mode", choices=MODES, default="global")
    p.add_argument("--match", type=int, default=2)
    p.add_argument("--mismatch", type=int, default=-1)
    p.add_argument("--gap", type=int, default=-2, help="linear gap penalty (global, local, linear-space)")
    p.add_argument("--gap-open", type=int, default=-3, help="affine gap open (affine mode)")
    p.add_argument("--gap-extend", type=int, default=-1, help="affine gap extend (affine mode)")
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--python", action="store_true", help="force the pure-Python backend")
    args = p.parse_args(argv)

    (name1, s1), (name2, s2) = read_sequence(args.seq1), read_sequence(args.seq2)
    aligner = SequenceAligner(args.match, args.mismatch, args.gap, args.gap_open, args.gap_extend,
                              use_cpp=not args.python)
    try:
        a1, a2, score = getattr(aligner, MODES[args.mode])(s1, s2)
    except ValueError as e:
        raise SystemExit(str(e))

    stats = alignment_stats(a1, a2)
    print(f"# {name1} ({len(s1)} bp) vs {name2} ({len(s2)} bp) | mode={args.mode} backend={aligner.backend}")
    print(f"# score={score} length={stats.length} identity={stats.identity:.1%} "
          f"mismatches={stats.mismatches} gaps={stats.gaps} gap_opens={stats.gap_opens}\n")
    print(format_alignment(a1, a2, args.width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
