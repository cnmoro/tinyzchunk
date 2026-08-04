"""Grid-search the chunker's inference parameters against the scenario matrix.

Reports macro F1 over the synthetic/noisy/augmented buckets alongside F1 on the
real held-out documents, so a setting is only chosen when it helps both.
"""
import argparse
import collections
import itertools
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from tinyzchunk.chunker import Chunker
from eval_matrix import EVAL_SETS, doc_metrics


def load(per_bucket):
    random.seed(0)
    buckets = collections.defaultdict(list)
    for tag, path in EVAL_SETS:
        if not os.path.exists(path):
            continue
        by_src = collections.defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            by_src[r.get("source", "?")].append(r)
        for src, rs in by_src.items():
            random.shuffle(rs)
            buckets[f"{tag}/{src}"] = rs[:per_bucket]
    ref = "data/heldout_labels_gen/labels.jsonl"
    if os.path.exists(ref):
        buckets["real/heldout"] = [json.loads(l) for l in open(ref)]
    return buckets


def score(c, buckets):
    per = {}
    for name, recs in buckets.items():
        f1 = [doc_metrics(c, r["text"], r.get("boundaries"))["f1"] for r in recs]
        per[name] = float(np.nanmean(f1)) if f1 else float("nan")
    macro = float(np.nanmean([v for k, v in per.items() if k != "real/heldout"]))
    return macro, per.get("real/heldout", float("nan")), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=12)
    ap.add_argument("--big", nargs="+", type=float, default=[0.35, 0.5, 0.65, 0.8])
    ap.add_argument("--min-chunk", nargs="+", type=int, default=[40, 100])
    ap.add_argument("--blend", nargs="+", type=float, default=[0.0, 0.15, 0.3])
    ap.add_argument("--save-best", default=None)
    args = ap.parse_args()

    buckets = load(args.per_bucket)
    print(f"{sum(len(v) for v in buckets.values())} docs across {len(buckets)} buckets\n")
    print(f"{'big':>6}{'minchunk':>10}{'blend':>7}{'macroF1':>10}{'heldoutF1':>11}")
    rows = []
    for big, mc, bl in itertools.product(args.big, args.min_chunk, args.blend):
        c = Chunker(big_threshold=big, small_threshold=big,
                    min_chunk_chars=mc, char_blend=bl)
        macro, held, per = score(c, buckets)
        rows.append((macro + held, macro, held, big, mc, bl, per))
        print(f"{big:>6}{mc:>10}{bl:>7}{macro:>10.4f}{held:>11.4f}", flush=True)

    rows.sort(reverse=True, key=lambda r: r[0])
    _, macro, held, big, mc, bl, per = rows[0]
    print(f"\nBEST big={big} min_chunk={mc} blend={bl} "
          f"-> macro {macro:.4f}, heldout {held:.4f}")
    worst = sorted(((v, k) for k, v in per.items() if np.isfinite(v)))[:10]
    print("weakest buckets at that setting:")
    for v, k in worst:
        print(f"  {k:<30}{v:.3f}")
    if args.save_best:
        json.dump({"big_threshold": big, "min_chunk_chars": mc,
                   "char_blend": bl, "macro_f1": macro, "heldout_f1": held},
                  open(args.save_best, "w"), indent=1)


if __name__ == "__main__":
    main()
