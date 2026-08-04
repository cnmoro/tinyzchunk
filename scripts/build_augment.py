"""Boundary-preserving degradation of already-labelled documents.

Every labelled document is split into its units (a unit starts at each labelled
boundary), each unit is passed through one or more text degradations, and the
units are concatenated again.  Because units are transformed independently, the
new boundary offsets are just the cumulative lengths -- the labels survive the
degradation for free.

This is the cheapest source of scenario coverage in the project: it converts the
existing corpus into accent-stripped, ALL-CAPS, punctuation-free, markup-free,
CRLF, tab-indented and page-furniture variants without a single GPU second.
"""
import argparse
import collections
import json
import os
import random
import re
import unicodedata

SOURCES = [
    "data/labels/labels.jsonl",
    "data/struct_labels/labels.jsonl",
    "data/synth_labels/labels.jsonl",
    "data/canarim_labels/labels.jsonl",
]

# ------------------------------------------------------------- unit transforms


def strip_accents(u):
    out = []
    for ch in u:
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        out.append(base if len(base) == 1 else ch)
    return "".join(out)


def to_upper(u):
    return "".join(c.upper() if len(c.upper()) == 1 else c for c in u)


def to_lower(u):
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in u)


def strip_markdown(u):
    lines = []
    for ln in u.split("\n"):
        ln = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", ln)
        ln = re.sub(r"__([^_]+)__", r"\1", ln)
        ln = re.sub(r"`([^`]+)`", r"\1", ln)
        ln = re.sub(r"^(\s*)#{1,6}\s+", r"\1", ln)
        ln = re.sub(r"^(\s*)[-*•]\s+", r"\1", ln)
        lines.append(ln)
    return "\n".join(lines)


def drop_punct(u):
    return re.sub(r"[.,;:!?\"'()\[\]]", "", u)


def asr_style(u):
    """Speech-recognition output: no case, no punctuation."""
    return drop_punct(to_lower(u))


def unicode_punct(u):
    u = u.replace(" - ", " – ").replace("...", "…")
    u = re.sub(r'"([^"]*)"', r"“\1”", u)
    u = re.sub(r"'([a-zA-Zà-ÿ]+)'", r"‘\1’", u)
    return u


def tabify(u):
    lines = []
    for ln in u.split("\n"):
        m = re.match(r"^( {2,})", ln)
        if m:
            ln = "\t" * (len(m.group(1)) // 2) + ln[len(m.group(1)):]
        lines.append(ln)
    return "\n".join(lines)


def crlf(u):
    return u.replace("\n", "\r\n")


def bullet_swap(u):
    b = random.choice(["*", "•", "‣", "–", "+"])
    return re.sub(r"^(\s*)[-*•]\s", rf"\1{b} ", u, flags=re.M)


def header_swap(u):
    """Turn ATX headings into plain / setext / ALL-CAPS headings."""
    style = random.choice(["plain", "setext", "upper", "numbered"])
    lines = u.split("\n")
    out = []
    for ln in lines:
        m = re.match(r"^(\s*)#{1,6}\s+(.*)$", ln)
        if not m:
            out.append(ln)
            continue
        ind, txt = m.group(1), m.group(2).rstrip("#").strip()
        if style == "setext":
            out.append(ind + txt)
            out.append(ind + random.choice("=-") * max(len(txt), 3))
        elif style == "upper":
            out.append(ind + to_upper(txt))
        elif style == "numbered":
            out.append(f"{ind}{random.randint(1, 9)}. {txt}")
        else:
            out.append(ind + txt)
    return "\n".join(out)


def rewrap(u):
    """Re-wrap paragraphs to a narrow width (PDF-extraction look)."""
    width = random.randint(42, 70)
    out = []
    for para in u.split("\n"):
        if len(para) <= width or not para.strip():
            out.append(para)
            continue
        words, cur, curlen = para.split(), [], 0
        wrapped = []
        for w in words:
            if cur and curlen + 1 + len(w) > width:
                wrapped.append(" ".join(cur))
                cur, curlen = [], 0
            cur.append(w)
            curlen += len(w) + 1
        if cur:
            wrapped.append(" ".join(cur))
        out.append("\n".join(wrapped))
    return "\n".join(out)


def squeeze_blanks(u):
    return re.sub(r"\n{2,}", "\n", u)


def pad_blanks(u):
    return u.rstrip("\n") + "\n\n\n"


def trailing_spaces(u):
    return "\n".join(ln + " " * random.randint(0, 3) for ln in u.split("\n"))


def nbsp(u):
    return re.sub(r" ", lambda m: " " if random.random() < 0.15 else " ", u)


UNIT_TRANSFORMS = [
    (strip_accents, 0.16), (to_upper, 0.05), (to_lower, 0.06),
    (strip_markdown, 0.16), (drop_punct, 0.06), (asr_style, 0.06),
    (unicode_punct, 0.12), (tabify, 0.06), (bullet_swap, 0.10),
    (header_swap, 0.14), (rewrap, 0.16), (squeeze_blanks, 0.08),
    (pad_blanks, 0.06), (trailing_spaces, 0.10), (nbsp, 0.06),
]

# ------------------------------------------------------- document-level layout

PAGE_HEAD_EN = ["Confidential", "Internal use only", "Company Handbook",
                "Annual Report 2024", "Draft - do not distribute"]
PAGE_HEAD_PT = ["Confidencial", "Uso interno", "Manual Institucional",
                "Relatório Anual 2024", "Minuta - não distribuir"]


def page_furniture(units, lang):
    """Insert page numbers / running headers between units, as PDF text does."""
    heads = PAGE_HEAD_PT if lang == "pt" else PAGE_HEAD_EN
    head = random.choice(heads)
    page = random.randint(1, 5)
    out = []
    acc = 0
    for i, u in enumerate(units):
        if i and acc > random.randint(500, 1400):
            style = random.random()
            if style < 0.4:
                u = f"\n{page}\n\n" + u
            elif style < 0.7:
                u = f"\n{head} | {page}\n\n" + u
            else:
                u = f"\nPage {page} of 12\n\n" + u
            page += 1
            acc = 0
        acc += len(u)
        out.append(u)
    return out


def renumber(units):
    """Prefix every unit with a running number, turning it into a list."""
    fmt = random.choice(["{}. ", "{}) ", "[{}] ", "{}- "])
    return [(fmt.format(i + 1) + u.lstrip("\n")) if i or True else u
            for i, u in enumerate(units)]


DOC_TRANSFORMS = [(page_furniture, 0.22), (None, 0.0)]


# --------------------------------------------------------------------- driver

def split_units(text, boundaries):
    bds = sorted({b for b in boundaries if 0 < b < len(text)})
    edges = [0] + bds + [len(text)]
    return [text[a:b] for a, b in zip(edges, edges[1:])]


def augment(rec, labels_from_record):
    import numpy as np
    text = rec["text"]
    bds = np.flatnonzero(labels_from_record(rec, "big")).tolist()
    units = split_units(text, bds)
    if len(units) < 2 and len(text) < 300:
        return None
    lang = rec.get("lang", "pt")

    picked = [f for f, p in UNIT_TRANSFORMS if random.random() < p]
    if not picked:
        picked = [random.choice([f for f, _ in UNIT_TRANSFORMS])]
    random.shuffle(picked)
    picked = picked[:3]

    units = [u for u in units if u.strip()]
    if not units:
        return None
    new_units = []
    for u in units:
        for f in picked:
            u = f(u)
        new_units.append(u)

    applied = [f.__name__ for f in picked]
    if random.random() < 0.22:
        new_units = page_furniture(new_units, lang)
        applied.append("page_furniture")
    elif random.random() < 0.10:
        new_units = renumber(new_units)
        applied.append("renumber")
    if random.random() < 0.10:
        new_units = [crlf(u) for u in new_units]
        applied.append("crlf")

    out, pos, new_bds = [], 0, []
    for i, u in enumerate(new_units):
        if i:
            new_bds.append(pos)
        out.append(u)
        pos += len(u)
    joined = "".join(out)
    if len(joined) < 80:
        return None
    return {"lang": lang, "source": "aug_" + rec.get("source", "?"),
            "text": joined, "boundaries": [b for b in new_bds if 0 < b < len(joined)],
            "transforms": applied}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--out", default="data/aug_labels/labels.jsonl")
    ap.add_argument("--variants", type=int, default=1,
                    help="augmented copies per source document")
    ap.add_argument("--limit", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    import sys
    sys.path.insert(0, ".")
    from tinyzchunk.labels import labels_from_record

    records = []
    for p in args.sources:
        try:
            recs = [json.loads(l) for l in open(p)]
        except FileNotFoundError:
            print(f"  (skipping missing {p})")
            continue
        print(f"  {p}: {len(recs)} docs")
        records.extend(recs)
    random.shuffle(records)

    out = []
    for rec in records:
        for _ in range(args.variants):
            a = augment(rec, labels_from_record)
            if a:
                out.append(a)
        if len(out) >= args.limit:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, d in enumerate(out):
            d["doc_id"] = i
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    tr = collections.Counter(t for d in out for t in d["transforms"])
    print(f"TOTAL {len(out)} augmented docs -> {args.out}")
    for k, v in sorted(tr.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
