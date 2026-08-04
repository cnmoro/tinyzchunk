"""Replace a document's labels with a different teacher's, matched by text.

Used to A/B one teacher against another on the same documents: the corpus, the
model and every other label set stay fixed, so any change in the scenario matrix
is attributable to the labels alone.

    python scripts/merge_labels.py --base data/labels/labels.jsonl \
        --override data/real_labels_gen/labels.jsonl \
        --out data/labels_mixed/labels.jsonl
"""
import argparse
import json
import os


def key(text):
    return " ".join(text.split())[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/labels/labels.jsonl")
    ap.add_argument("--override", default="data/real_labels_gen/labels.jsonl")
    ap.add_argument("--out", default="data/labels_mixed/labels.jsonl")
    args = ap.parse_args()

    over = {}
    for line in open(args.override):
        r = json.loads(line)
        if r.get("boundaries"):
            over[key(r["text"])] = r["boundaries"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_over = n_keep = 0
    with open(args.out, "w") as f:
        for line in open(args.base):
            r = json.loads(line)
            k = key(r["text"])
            if k in over:
                # drop the log-prob arrays so labels_from_record uses the
                # explicit boundaries from the overriding teacher
                r.pop("big_logp", None)
                r.pop("small_logp", None)
                r.pop("cont_logp", None)
                r["boundaries"] = over[k]
                r["source"] = r.get("source", "?")
                n_over += 1
            else:
                n_keep += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n_over} documents relabelled, {n_keep} kept as-is -> {args.out}")
    if n_over == 0:
        print("WARNING: nothing matched; check that the two files share documents")


if __name__ == "__main__":
    main()
