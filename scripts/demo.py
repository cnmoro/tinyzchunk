"""Qualitative demo: chunk a few sample documents with the tiny chunker."""
import json
import random
import sys

sys.path.insert(0, ".")
from tinyzchunk import Chunker

random.seed(3)


def main():
    records = [json.loads(l) for l in open("data/labels/labels.jsonl")]
    picked = {}
    for r in records:
        picked.setdefault((r["lang"], r["source"]), r)
    c = Chunker()
    for key in [("en", "news"), ("pt", "wiki"), ("en", "legal"), ("pt", "triplets"), ("en", "stories")]:
        r = picked[key]
        print("=" * 90)
        print(f"[{key}] {len(r['text'])} chars  ->  {len(c.chunk(r['text']))} chunks")
        for i, ch in enumerate(c.chunk(r["text"])):
            print(f"  --- chunk {i + 1} ({len(ch)} chars) ---")
            print("  " + ch.replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
