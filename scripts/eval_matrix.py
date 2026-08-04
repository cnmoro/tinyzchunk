"""Scenario regression matrix for the chunker.

Reports, per scenario bucket, boundary agreement with the reference labels plus
the failure modes that matter in a RAG pipeline (fragment chunks, oversized
chunks, undersized chunks).  Every bucket is held out from training.

The point of this harness is lesson-3 insurance: adding training data for one
pattern routinely breaks another, so no weight change ships unless the whole
matrix is checked.  A final section asserts hard invariants that must hold for
any input at all.

    python scripts/eval_matrix.py                 # full matrix
    python scripts/eval_matrix.py --quick         # buckets only, no invariants
    python scripts/eval_matrix.py --baseline out.json --compare prev.json
"""
import argparse
import collections
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from tinyzchunk.chunker import Chunker

EVAL_SETS = [
    ("synth", "data/synth_eval/labels.jsonl"),
    ("aug", "data/aug_eval/labels.jsonl"),
    ("noisy", "data/noisy_eval/labels.jsonl"),
]


def boundary_prf(pred, gold, tol=3):
    if not len(pred) and not len(gold):
        return 1.0, 1.0, 1.0
    if not len(pred) or not len(gold):
        return 0.0, 0.0, 0.0
    used = [False] * len(gold)
    matched = 0
    for p in pred:
        for gi, g in enumerate(gold):
            if not used[gi] and abs(p - g) <= tol:
                matched += 1
                used[gi] = True
                break
    pr = matched / len(pred)
    rc = matched / len(gold)
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def chunk_offsets(text, chunks):
    """Start offset of each chunk after the first (i.e. predicted boundaries)."""
    pos, offs = 0, []
    for ch in chunks:
        s = text.find(ch, pos)
        if s < 0:
            s = pos
        offs.append(s)
        pos = s + len(ch)
    return offs[1:]


def doc_metrics(c, text, gold):
    chunks = c.chunk(text)
    pred = chunk_offsets(text, chunks)
    f1, pr, rc = boundary_prf(pred, gold) if gold is not None else (np.nan,) * 3
    sizes = [len(x) for x in chunks] or [0]
    # the first chunk inherits the document's own opening, which in sliced
    # corpora is often mid-sentence: only later chunks can be true fragments
    frag = sum(1 for x in chunks[1:]
               if x[:1].islower() and len(x) > 2) / max(len(chunks) - 1, 1)
    over = sum(1 for s in sizes if s > c.max_chunk_chars) / len(sizes)
    tiny = sum(1 for s in sizes if s < c.min_chunk_chars) / len(sizes)
    return dict(f1=f1, precision=pr, recall=rc, n_chunks=len(chunks),
                median=float(np.median(sizes)), p95=float(np.percentile(sizes, 95)),
                frag=frag, over=over, tiny=tiny)


def agg(rows):
    keys = ("f1", "precision", "recall", "frag", "over", "tiny", "median", "p95")
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out["docs"] = len(rows)
    out["chunks"] = int(sum(r["n_chunks"] for r in rows))
    return out


# ------------------------------------------------------------------ invariants

def check_invariants(c, samples):
    """Hard properties that must hold for any input whatsoever."""
    fails = collections.Counter()
    examples = {}

    def fail(name, detail):
        fails[name] += 1
        examples.setdefault(name, detail)

    for text in samples:
        chunks = c.chunk(text)
        for ch in chunks:
            if ch not in text:
                fail("chunk_not_substring", repr(ch[:60]))
                break
        # no chunk may begin in the middle of a word
        for ch in chunks:
            i = text.find(ch)
            if i > 0 and text[i - 1].isalnum() and text[i].isalnum():
                fail("mid_word_cut", repr(text[max(0, i - 12):i + 12]))
                break
        if any(len(ch) > c.max_chunk_chars for ch in chunks):
            fail("oversized", max(len(ch) for ch in chunks))
        # dropping content is the one thing a chunker must never do; compare
        # non-whitespace characters, since chunking legitimately changes
        # where whitespace falls
        joined = "".join("".join(chunks).split())
        want = "".join(text.split())
        if joined != want:
            fail("content_lost", f"{len(joined)} vs {len(want)} chars")
        # CRLF must behave exactly like LF
        got = [x.replace("\r\n", "\n").strip() for x in c.chunk(
            text.replace("\n", "\r\n"))]
        if got != [x.strip() for x in chunks]:
            fail("crlf_mismatch", f"{len(got)} vs {len(chunks)} chunks")

    edge_cases = [
        "", " ", "\n", "\n\n\n", "\r\n\r\n", "a", "a b c",
        "x" * 5000, "word " * 2000, "\n".join(["line"] * 500),
        "Sentence one. Sentence two. " * 200,
        "Título\n" + "=" * 6 + "\n\nTexto.\n", "﻿BOM start\nsecond line\n",
        "# Header\n\n```\ncode\nmore code\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        "\t\tindented\n\t\tmore\n", "🎉 emoji doc 🎉\n\nsecond unit here.\n",
    ]
    for t in edge_cases:
        try:
            out = c.chunk(t)
            if t.strip() and "".join("".join(out).split()) != "".join(t.split()):
                fail("edge_content_lost", repr(t[:40]))
        except Exception as e:  # noqa: BLE001
            fail("edge_exception", f"{type(e).__name__}: {e} on {t[:40]!r}")
    return fails, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="tinyzchunk/weights.npz")
    ap.add_argument("--line-weights", default="tinyzchunk/line_weights.npz")
    ap.add_argument("--big-threshold", type=float, default=0.70)
    ap.add_argument("--small-threshold", type=float, default=0.70)
    ap.add_argument("--min-chunk-chars", type=int, default=100)
    ap.add_argument("--max-chunk-chars", type=int, default=2500)
    ap.add_argument("--char-blend", type=float, default=0.15)
    ap.add_argument("--per-doc-limit", type=int, default=220,
                    help="max docs sampled per bucket (keeps the run quick)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    random.seed(0)
    c = Chunker(weights_path=args.weights, line_weights_path=args.line_weights,
                big_threshold=args.big_threshold,
                small_threshold=args.small_threshold,
                min_chunk_chars=args.min_chunk_chars,
                max_chunk_chars=args.max_chunk_chars,
                char_blend=args.char_blend)

    buckets = collections.defaultdict(list)
    samples = []

    for tag, path in EVAL_SETS:
        if not os.path.exists(path):
            print(f"  (missing {path})")
            continue
        recs = [json.loads(l) for l in open(path)]
        by_src = collections.defaultdict(list)
        for r in recs:
            by_src[r.get("source", "?")].append(r)
        for src, rs in by_src.items():
            random.shuffle(rs)
            for r in rs[:args.per_doc_limit]:
                buckets[f"{tag}/{src}"].append(
                    doc_metrics(c, r["text"], r.get("boundaries")))
            samples.extend(r["text"] for r in rs[:3])

    # real held-out documents, matched by text (never by filename order)
    ref_path = "data/heldout_labels_gen/labels.jsonl"
    if os.path.exists(ref_path):
        for r in (json.loads(l) for l in open(ref_path)):
            buckets["real/heldout"].append(
                doc_metrics(c, r["text"], r.get("boundaries")))
            samples.append(r["text"])

    # unlabelled generalization corpus: structural metrics only
    hf_path = "data/hf_eval_corpus.jsonl"
    if os.path.exists(hf_path):
        for r in (json.loads(l) for l in open(hf_path)):
            buckets["real/hf_generalization"].append(
                doc_metrics(c, r["text"], None))
            samples.append(r["text"])

    print(f"\n{'bucket':<28}{'docs':>5}{'F1':>7}{'P':>7}{'R':>7}"
          f"{'frag%':>7}{'over%':>7}{'tiny%':>7}{'med':>6}{'p95':>7}")
    print("-" * 88)
    results = {}
    for name in sorted(buckets):
        a = agg(buckets[name])
        results[name] = a
        f1 = f"{a['f1']:.3f}" if np.isfinite(a["f1"]) else "  -  "
        pr = f"{a['precision']:.3f}" if np.isfinite(a["precision"]) else "  -  "
        rc = f"{a['recall']:.3f}" if np.isfinite(a["recall"]) else "  -  "
        print(f"{name:<28}{a['docs']:>5}{f1:>7}{pr:>7}{rc:>7}"
              f"{100*a['frag']:>7.1f}{100*a['over']:>7.1f}{100*a['tiny']:>7.1f}"
              f"{a['median']:>6.0f}{a['p95']:>7.0f}")

    labelled = [v["f1"] for k, v in results.items() if np.isfinite(v["f1"])]
    macro = float(np.mean(labelled)) if labelled else float("nan")
    frag = float(np.mean([v["frag"] for v in results.values()]))
    over = float(np.mean([v["over"] for v in results.values()]))
    print("-" * 88)
    print(f"MACRO F1 {macro:.4f} over {len(labelled)} labelled buckets   "
          f"| mean frag {100*frag:.2f}%  mean oversized {100*over:.2f}%")

    if not args.quick:
        random.shuffle(samples)
        fails, examples = check_invariants(c, samples[:180])
        print("\nINVARIANTS")
        if not fails:
            print("  all passed")
        for k, v in fails.most_common():
            print(f"  FAIL {k:<22} x{v:<4} e.g. {examples[k]}")
        results["_invariants"] = dict(fails)

    results["_macro"] = macro
    if args.save:
        json.dump(results, open(args.save, "w"), indent=1)
        print("\nsaved:", args.save)
    if args.compare and os.path.exists(args.compare):
        prev = json.load(open(args.compare))
        print(f"\nCOMPARED WITH {args.compare} (delta F1, worse first)")
        deltas = []
        for k, v in results.items():
            if k.startswith("_") or k not in prev:
                continue
            if np.isfinite(v.get("f1", np.nan)) and np.isfinite(prev[k].get("f1", np.nan)):
                deltas.append((v["f1"] - prev[k]["f1"], k))
        for d, k in sorted(deltas)[:12]:
            flag = "REGRESSION" if d < -0.02 else ""
            print(f"  {k:<28}{d:+.3f}  {flag}")
        print(f"  MACRO {macro - prev.get('_macro', float('nan')):+.4f}")


if __name__ == "__main__":
    main()
