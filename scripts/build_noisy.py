"""Build noisy-text training/eval data by applying realistic PDF/OCR noise to
clean documents while tracking boundary positions through the noise.

Noise types simulated (as seen in real-world PDF extractions):
  - narrow line wrapping, sometimes splitting words mid-word
  - page numbers + form feeds (\x0c) inserted between units
  - OCR-style mid-word spaces ("Polí ca", "par cipantes")
  - stray digits / artifacts

Each clean unit (whose start is a boundary) stays a unit; the noise is applied
INSIDE units so the model learns that wrapped lines / page numbers / mangled
words are not boundaries.
"""
import json
import os
import random
import sys

sys.path.insert(0, ".")

random.seed(11)

MANGLES = [("ti", " "), ("rn", " "), ("ci", " "), ("en", " "), ("on", " "),
           ("co", " "), ("im", " "), ("ne", " "), ("st", " "), ("pr", " ")]


def mangle_word(w):
    if len(w) < 7 or random.random() > 0.03:
        return w
    for pat, rep in MANGLES:
        if pat in w.lower():
            i = w.lower().index(pat)
            return w[:i + 1] + " " + w[i + 2:]
    return w


def noisy_wrap(text, width):
    words = text.split()
    lines, cur, curlen = [], [], 0
    for w in words:
        if cur and curlen + 1 + len(w) > width:
            if random.random() < 0.15 and cur and len(cur[-1]) > 6:
                last = cur[-1]
                cut = random.randint(3, len(last) - 3)
                cur[-1] = last[:cut]
                lines.append(" ".join(cur))
                cur = [last[cut:]]
                curlen = len(last[cut:])
                continue
            lines.append(" ".join(cur))
            cur, curlen = [], 0
        cur.append(mangle_word(w))
        curlen += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def apply_noise(text, boundaries):
    """Split into units at boundaries, add noise inside units, return
    (noisy_text, noisy_boundaries).  Units are separated by blank lines."""
    boundaries = sorted(set(b for b in boundaries if 0 < b < len(text)))
    starts = [0] + boundaries
    ends = boundaries + [len(text)]
    out = []
    out_pos = 0
    new_bds = []
    page_counter = 0
    for si, (s, e) in enumerate(zip(starts, ends)):
        unit = text[s:e].strip("\n")
        if not unit:
            continue
        if si > 0:
            out.append("\n\n")
            out_pos += 2
            new_bds.append(out_pos)
        wrapped = noisy_wrap(unit, random.randint(45, 68))
        if random.random() < 0.3:
            wrapped = wrapped + " " + str(random.randint(1, 40))
        out.append(wrapped)
        out_pos += len(wrapped)
        page_counter += len(wrapped)
        if page_counter > random.randint(800, 1200):
            out.append("\n\n" + str(random.randint(1, 50)) + "\x0c\n\n")
            out_pos += len("\n\n" + str(random.randint(1, 50)) + "\x0c\n\n")
            page_counter = 0
    noisy = "".join(out)
    return noisy, [b for b in new_bds if 0 < b < len(noisy)]


def main():
    from tinyzchunk.labels import labels_from_record
    import numpy as np
    records = [json.loads(l) for l in open("data/labels/labels.jsonl")]
    # also add structured docs (Q&A scripts, schedules, FAQs, bios) in noisy form
    for path in ["data/struct_labels/labels.jsonl",
                 "data/synth_labels/labels.jsonl"]:
        try:
            records.extend(json.loads(l) for l in open(path))
        except FileNotFoundError:
            print("  (skipping missing", path + ")")
    random.shuffle(records)
    train_out, eval_out = [], []
    for i, rec in enumerate(records[:3000]):
        text = rec["text"]
        if len(text) < 200:
            continue
        bds = np.flatnonzero(labels_from_record(rec, "big"))
        noisy, nbds = apply_noise(text, bds.tolist())
        if not nbds:
            continue
        doc = {"doc_id": i, "lang": rec.get("lang", "pt"),
               "source": "noisy_" + rec.get("source", "?"),
               "text": noisy, "boundaries": nbds}
        if len(train_out) < 2400:
            train_out.append(doc)
        else:
            eval_out.append(doc)
    os.makedirs("data/noisy_train", exist_ok=True)
    os.makedirs("data/noisy_eval", exist_ok=True)
    for path, docs in [(os.path.join("data/noisy_train", "labels.jsonl"), train_out),
                       (os.path.join("data/noisy_eval", "labels.jsonl"), eval_out)]:
        with open(path, "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    import collections
    srcs = collections.Counter(d["source"] for d in train_out)
    print(f"noisy train {len(train_out)} docs, eval {len(eval_out)}")
    print("train sources:", dict(srcs))


if __name__ == "__main__":
    main()
