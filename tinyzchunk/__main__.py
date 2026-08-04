"""CLI: chunk a file (or stdin) with tinyzchunk."""
import argparse
import sys

from tinyzchunk import Chunker


def main():
    ap = argparse.ArgumentParser(description="GPU-free chunking distilled from zChunk")
    ap.add_argument("input", nargs="?", default="-", help="file to chunk (default: stdin)")
    ap.add_argument("--big-threshold", type=float, default=0.50)
    ap.add_argument("--small-threshold", type=float, default=0.50)
    ap.add_argument("--max-chunk-chars", type=int, default=2500)
    ap.add_argument("--min-chunk-chars", type=int, default=100)
    ap.add_argument("--char-blend", type=float, default=0.15,
                    help="weight of the char model in the line score")
    ap.add_argument("--json", action="store_true", help="emit a JSON array")
    ap.add_argument("--separator", default="\n---\n")
    ap.add_argument("--show-boundaries", action="store_true")
    args = ap.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = open(args.input).read()

    c = Chunker(big_threshold=args.big_threshold, small_threshold=args.small_threshold,
                max_chunk_chars=args.max_chunk_chars, min_chunk_chars=args.min_chunk_chars,
                char_blend=args.char_blend)
    if args.show_boundaries:
        for pos, kind in c.boundaries(text):
            print(f"{kind}\t{pos}\t...{text[max(0,pos-30):pos+10]!r}".replace("\n", "\\n"))
    elif args.json:
        import json
        print(json.dumps(c.chunk(text), ensure_ascii=False, indent=1))
    else:
        for ch in c.chunk(text):
            print(ch)
            print(args.separator, end="")


if __name__ == "__main__":
    main()
