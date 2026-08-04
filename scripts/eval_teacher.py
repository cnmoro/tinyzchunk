"""Score an LLM teacher against documents whose boundaries are known.

Teacher quality is normally invisible: you can only compare one teacher with
another, and two teachers labelling the same documents agree surprisingly badly
(F1 ~0.25 in this project), so agreement says little about correctness.

The synthetic corpus solves that. Those documents are generated from a known
structure, so their boundaries are ground truth, not opinion. Running a teacher
over them measures the thing that actually matters -- does this model put splits
where they belong -- and makes "is a bigger teacher worth it?" an experiment
rather than an assumption.

Reported per model:
  boundary F1/P/R  against the true boundaries
  fidelity         generated length / original length (verbatim copying)
  markers          how many splits it proposed vs how many exist

    python scripts/eval_teacher.py --model Qwen/Qwen3.5-4B --limit 40
"""
import argparse
import collections
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def boundary_prf(pred, gold, tol=8):
    if not len(pred) and not len(gold):
        return 1.0, 1.0, 1.0
    if not len(pred) or not len(gold):
        return 0.0, 0.0, 0.0
    used = [False] * len(gold)
    matched = 0
    for p in sorted(pred):
        for gi, g in enumerate(gold):
            if not used[gi] and abs(p - g) <= tol:
                matched += 1
                used[gi] = True
                break
    pr = matched / len(pred)
    rc = matched / len(gold)
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def load_docs(args):
    docs = [json.loads(l) for l in open(args.corpus)]
    docs = [d for d in docs
            if 300 < len(d["text"]) <= args.max_chars and d.get("boundaries")]
    random.Random(args.seed).shuffle(docs)
    return docs[:args.limit]


def run_logprob(args):
    """Score the zChunk log-probability teacher on the same documents."""
    from teacher import Teacher
    from tinyzchunk.labels import detect_boundaries, snap_positions

    docs = load_docs(args)
    print(f"logprob teacher: {len(docs)} documents from {args.corpus}\n")
    teacher = Teacher()
    rows, by_src = [], collections.defaultdict(list)
    t0 = time.time()
    for i, d in enumerate(docs):
        text, gold = d["text"], d["boundaries"]
        try:
            big, _small, _cont = teacher.score(text)
        except Exception as e:                                    # noqa: BLE001
            print(f"  [{i}] scoring failed: {type(e).__name__}: {e}")
            continue
        pred = snap_positions(text, detect_boundaries(big))
        f1, pr, rc = boundary_prf(pred, gold)
        rows.append(dict(source=d.get("source", "?"), f1=f1, precision=pr, recall=rc,
                         fidelity=1.0, markers=len(pred), gold=len(gold)))
        by_src[d.get("source", "?")].append(f1)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(docs)}] running F1 "
                  f"{np.mean([r['f1'] for r in rows]):.3f}", flush=True)
    report("logprob (zChunk split-token)", rows, by_src, t0, args)


def report(name, rows, by_src, t0, args):
    if not rows:
        raise SystemExit("no documents scored")
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("f1", "precision", "recall", "fidelity")}
    tot_m = sum(r["markers"] for r in rows)
    tot_g = sum(r["gold"] for r in rows)
    print(f"\n=== {name} ===")
    print(f"  boundary F1   {agg['f1']:.3f}   (P {agg['precision']:.3f} "
          f"R {agg['recall']:.3f})")
    print(f"  copy fidelity {agg['fidelity']:.3f}")
    print(f"  markers       {tot_m} proposed vs {tot_g} true "
          f"({tot_m/max(tot_g,1):.2f}x)")
    print(f"  docs {len(rows)}, {(time.time()-t0)/len(rows):.1f}s/doc")
    print("\n  weakest families:")
    for src, f1s in sorted(by_src.items(), key=lambda kv: np.mean(kv[1]))[:6]:
        print(f"    {src:<20}{np.mean(f1s):.3f}  (n={len(f1s)})")
    if args.out:
        json.dump({"model": name, "agg": agg, "rows": rows,
                   "markers": tot_m, "gold_markers": tot_g},
                  open(args.out, "w"), indent=1)
        print("\nsaved:", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to teacher_gen.DEFAULT_MODEL")
    ap.add_argument("--corpus", default="data/synth_eval/labels.jsonl")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2200,
                    help="only short docs, so sectioning never confounds the score")
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--quantize-head", action="store_true")
    ap.add_argument("--mode", choices=("generation", "logprob"), default="generation",
                    help="generation = 【SPLIT】 markers; logprob = the zChunk "
                         "split-token probability trick")
    args = ap.parse_args()

    if args.mode == "logprob":
        return run_logprob(args)

    from teacher_gen import (MARK, align_boundaries, generate_chunked,
                             load_model, DEFAULT_MODEL)
    model_name = args.model or DEFAULT_MODEL

    docs = [json.loads(l) for l in open(args.corpus)]
    docs = [d for d in docs
            if 300 < len(d["text"]) <= args.max_chars and d.get("boundaries")]
    random.Random(args.seed).shuffle(docs)
    docs = docs[:args.limit]
    print(f"{model_name}: {len(docs)} documents from {args.corpus}\n")

    tok, model = load_model(model_name, quantize_head=args.quantize_head)
    rows = []
    by_src = collections.defaultdict(list)
    t0 = time.time()
    for i, d in enumerate(docs):
        text, gold = d["text"], d["boundaries"]
        try:
            gen = generate_chunked(tok, model, text, args.max_tokens)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [{i}] generation failed: {type(e).__name__}: {e}")
            continue
        n_mark = gen.count(MARK)
        fidelity = len(gen.replace(MARK, "")) / max(len(text), 1)
        pred = align_boundaries(text, gen) if n_mark else []
        f1, pr, rc = boundary_prf(pred, gold)
        rows.append(dict(source=d.get("source", "?"), f1=f1, precision=pr, recall=rc,
                         fidelity=fidelity, markers=n_mark, gold=len(gold)))
        by_src[d.get("source", "?")].append(f1)
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(docs)}] running F1 "
                  f"{np.mean([r['f1'] for r in rows]):.3f} "
                  f"({el/(i+1):.1f}s/doc)", flush=True)

    if not rows:
        raise SystemExit("no documents scored")
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("f1", "precision", "recall", "fidelity")}
    tot_m = sum(r["markers"] for r in rows)
    tot_g = sum(r["gold"] for r in rows)
    print(f"\n=== {model_name} ===")
    print(f"  boundary F1   {agg['f1']:.3f}   (P {agg['precision']:.3f} "
          f"R {agg['recall']:.3f})")
    print(f"  copy fidelity {agg['fidelity']:.3f}   (1.000 = perfect verbatim)")
    print(f"  markers       {tot_m} proposed vs {tot_g} true "
          f"({tot_m/max(tot_g,1):.2f}x)")
    print(f"  docs {len(rows)}, {(time.time()-t0)/len(rows):.1f}s/doc")
    print("\n  weakest families:")
    for src, f1s in sorted(by_src.items(), key=lambda kv: np.mean(kv[1]))[:6]:
        print(f"    {src:<20}{np.mean(f1s):.3f}  (n={len(f1s)})")

    if args.out:
        json.dump({"model": model_name, "agg": agg, "rows": rows,
                   "markers": tot_m, "gold_markers": tot_g},
                  open(args.out, "w"), indent=1)
        print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
