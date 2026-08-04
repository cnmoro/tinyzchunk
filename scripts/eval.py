"""Evaluate the distilled chunker against the teacher and baselines."""
import argparse
import json
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from tinyzchunk.chunker import Chunker
from tinyzchunk.labels import boundary_labels


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


def fixed_boundaries(text, size=500):
    return list(range(size, len(text), size))


def sentence_boundaries(text, target=500):
    """Split on sentence terminators, group sentences into ~target-char chunks."""
    bounds = [0]
    i = 0
    while i < len(text):
        nxt = i + target
        seg = text[i:nxt]
        ends = [m.end() - 1 for m in re.finditer(r"[.!?;:]", seg)]
        if ends:
            i += max(ends) + 1
        else:
            i = nxt
        if i < len(text):
            bounds.append(i)
    return bounds[1:]


def paragraph_boundaries(text):
    return [m.end() - 1 for m in re.finditer(r"\n\s*\n", text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/labels.jsonl")
    ap.add_argument("--weights", default="tinyzchunk/weights.npz")
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--big-threshold", type=float, default=0.40)
    ap.add_argument("--small-threshold", type=float, default=0.55)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.labels)]
    rng = np.random.default_rng(7)
    rng.shuffle(records)
    val = records[: args.n_val]

    c = Chunker(weights_path=args.weights, big_threshold=args.big_threshold,
                small_threshold=args.small_threshold)

    rows = []
    for rec in val:
        text = rec["text"]
        tb = np.flatnonzero(boundary_labels(np.asarray(rec["big_logp"])))
        ts = np.flatnonzero(boundary_labels(np.asarray(rec["small_logp"])))
        sb = np.array([p for p, _ in c.boundaries(text)], dtype=np.int64)
        fb = np.array(fixed_boundaries(text), dtype=np.int64)
        sentb = np.array(sentence_boundaries(text), dtype=np.int64)
        pb = np.array(paragraph_boundaries(text), dtype=np.int64)
        rows.append({
            "doc": rec["doc_id"], "lang": rec["lang"], "src": rec["source"], "n": len(text),
            "teacher_big": len(tb), "student": len(sb),
            "f1_student": boundary_f1(sb, tb),
            "f1_fixed": boundary_f1(fb, tb),
            "f1_sent": boundary_f1(sentb, tb),
            "f1_para": boundary_f1(pb, tb),
        })

    def mean(k):
        return np.mean([r[k] for r in rows])

    print(f"{'method':<10} {'F1-big vs teacher':>18}")
    print(f"{'student':<10} {mean('f1_student'):>18.3f}")
    print(f"{'sentence':<10} {mean('f1_sent'):>18.3f}")
    print(f"{'paragraph':<10} {mean('f1_para'):>18.3f}")
    print(f"{'fixed500':<10} {mean('f1_fixed'):>18.3f}")
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r["f1_student"])
    for lang, f1s in by_lang.items():
        print(f"student F1 [{lang}]: {np.mean(f1s):.3f} (n={len(f1s)})")

    # chunk-size sanity
    sizes = []
    for rec in val:
        for ch in c.chunk(rec["text"]):
            sizes.append(len(ch))
    sizes = np.array(sizes)
    print(f"\nchunk sizes: mean={sizes.mean():.0f} p50={np.median(sizes):.0f} "
          f"max={sizes.max():.0f} min={sizes.min():.0f} n={len(sizes)}")


if __name__ == "__main__":
    main()
