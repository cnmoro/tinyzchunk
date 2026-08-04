"""Evaluate the chunker on the held-out hold-out documents.

The 28 hold-out files are NEVER used in training.  Two judgments are produced:
  1) objective boundary agreement vs a generation-teacher reference (run
     scripts/teacher_gen.py --in data/heldout_corpus.jsonl --out ... first)
  2) a human-readable chunk dump under data/heldout_out/ for manual review.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from tinyzchunk.chunker import Chunker
from tinyzchunk.labels import positions_to_labels


def boundary_f1(pred, gold, tol=3):
    if len(pred) == 0 and len(gold) == 0:
        return 1.0
    if len(pred) == 0 or len(gold) == 0:
        return 0.0
    matched = 0
    used = [False] * len(gold)
    for p in pred:
        for gi, g in enumerate(gold):
            if not used[gi] and abs(p - g) <= tol:
                matched += 1
                used[gi] = True
                break
    pr = matched / len(pred)
    rc = matched / len(gold)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/heldout")
    ap.add_argument("--ref", default="data/heldout_labels_gen/labels.jsonl",
                    help="generation-teacher reference labels (optional)")
    ap.add_argument("--weights", default="tinyzchunk/weights.npz")
    ap.add_argument("--out-dir", default="data/heldout_out")
    ap.add_argument("--big-threshold", type=float, default=0.50)
    ap.add_argument("--small-threshold", type=float, default=0.50)
    ap.add_argument("--min-chunk-chars", type=int, default=100)
    ap.add_argument("--max-chunk-chars", type=int, default=2500)
    args = ap.parse_args()

    # match reference records by TEXT, never by filename order: adding or
    # renaming a document must not silently shift every label
    ref = {}
    if os.path.exists(args.ref):
        for r in [json.loads(l) for l in open(args.ref)]:
            ref[" ".join(r["text"].split())[:400]] = r["boundaries"]

    os.makedirs(args.out_dir, exist_ok=True)
    c = Chunker(weights_path=args.weights, big_threshold=args.big_threshold,
                small_threshold=args.small_threshold,
                min_chunk_chars=args.min_chunk_chars,
                max_chunk_chars=args.max_chunk_chars)

    files = sorted(os.listdir(args.docs))
    f1s = []
    for name in files:
        text = open(os.path.join(args.docs, name)).read().strip()
        if not text:
            continue
        chunks = c.chunk(text)
        # align chunk char offsets by searching in the text
        pos = 0
        pred = []
        for ch in chunks:
            s = text.find(ch, pos)
            if s < 0:
                s = pos
            pos = s + len(ch)
            pred.append(pos)
        sizes = [len(x) for x in chunks]
        key = " ".join(text.split())[:400]
        f1 = np.nan
        if key in ref:
            gold = ref[key]
            f1 = boundary_f1(np.array(pred, dtype=np.int64), np.array(gold, dtype=np.int64))
            f1s.append(f1)
        # dump for manual review
        with open(os.path.join(args.out_dir, name + ".chunks.txt"), "w") as f:
            for i, ch in enumerate(chunks):
                f.write(f"CHUNK {i+1} ({len(ch)} chars)\n{ch}\n\n---\n\n")
        print(f"{name[:48]:<50} {len(text):>6}  {len(chunks):>3} chunks  sizes={min(sizes) if sizes else 0}-{np.median(sizes):.0f}-{max(sizes) if sizes else 0}"
              + (f"  F1={f1:.3f}" if np.isfinite(f1) else "  (no ref)"))
    if f1s:
        print(f"\nMEAN boundary F1 vs generation teacher: {np.mean(f1s):.3f}  (n={len(f1s)})")


if __name__ == "__main__":
    main()
