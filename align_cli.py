"""
Command-line pairwise aligner.

    seqalign GATTACA GCATGCU
    seqalign query.fasta reference.fasta --mode local
    python align_cli.py ACGTTTTACG ACGACG --mode affine --gap-open -5 --gap-extend -1
"""
import argparse
import os
import sys

from aligner import SequenceAligner, alignment_stats, format_alignment

MODES = {
    "global": "needleman_wunsch",
    "local": "smith_waterman",
    "affine": "gotoh",
    "linear-space": "hirschberg",
}


def read_sequence(arg: str) -> tuple:
    """Return (name, sequence) from a FASTA/plain-text file path, or treat arg as a literal sequence."""
    if not os.path.isfile(arg):
        return "seq", arg.strip()
    name, chunks = os.path.basename(arg), []
    with open(arg) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if chunks:
                    break  # first record only
                name = line[1:].split()[0] if len(line) > 1 else name
            elif line:
                chunks.append(line)
    if not chunks:
        raise SystemExit(f"{arg}: no sequence found")
    return name, "".join(chunks)


def main(argv=None):
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
