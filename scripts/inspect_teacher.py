"""Quick sanity check: run the teacher on a few docs and inspect the split signal."""
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/nvme1tb/Carlo/custom-chunker/scripts")
from teacher import Teacher


def show(text, big, small, top_n=15, width=45):
    n = len(text)
    print("=" * 100)
    order = np.argsort(big)[::-1]
    printed = 0
    for idx in order:
        if printed >= top_n:
            break
        if big[idx] < -8:
            continue
        ctx = text[max(0, idx - width): idx + 2]
        ctx = ctx.replace("\n", "\\n")
        print(f"  {idx:>5}  big={big[idx]:+.2f} small={small[idx]:+.2f}  ...{ctx}...")
        printed += 1


def main():
    t = Teacher()
    docs = [json.loads(l) for l in open("/mnt/nvme1tb/Carlo/custom-chunker/data/corpus.jsonl")]
    # pick one of each source
    picked = {}
    for d in docs:
        picked.setdefault((d["lang"], d["source"]), d)
    for key in [("en", "stories"), ("pt", "wiki"), ("en", "legal"), ("pt", "triplets"), ("en", "news")]:
        d = picked[key]
        print("\n\nDOC:", key, "len", len(d["text"]))
        print(d["text"][:200].replace("\n", "\\n"))
        big, small, cont = t.score(d["text"])
        show(d["text"], big, small)


if __name__ == "__main__":
    main()
