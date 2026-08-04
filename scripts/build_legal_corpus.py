"""Build a real legal-document training corpus from HuggingFace datasets.

Real dense legal text (FCC regulations, US government regulations) teaches the
chunker the 'dense legal document' pattern that is found in real-world legal corpora:
section headers are units, but the dense prose/lists in between are not.
"""
import json
import re

from datasets import load_dataset


def clean(t):
    t = re.sub(r"\r\n?", "\n", str(t))
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def section_doc(text, target=1900):
    """Split a long legal doc into ~target-char docs at paragraph/sentence breaks."""
    paras = [p for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 60]
    if not paras:
        paras = [text]
    out, cur, curlen = [], [], 0
    for p in paras:
        if curlen + len(p) > target and cur:
            out.append("\n\n".join(cur))
            cur, curlen = [], 0
        cur.append(p)
        curlen += len(p) + 2
    if cur:
        out.append("\n\n".join(cur))
    return [d for d in out if 400 <= len(d) <= 2600]


def main():
    docs = []
    # FCC regulations
    n_fcc = 0
    fcc = load_dataset("lucyd/fcc-regulations", split="train", streaming=True)
    for ex in fcc:
        if n_fcc >= 35:
            break
        for sec in section_doc(clean(ex["Content"])):
            if n_fcc >= 35:
                break
            docs.append({"lang": "en", "source": "legal_fcc", "text": sec})
            n_fcc += 1
    print("fcc:", n_fcc)

    # common-pile regulations
    n_reg = 0
    reg = load_dataset("common-pile/regulations", split="train", streaming=True)
    for ex in reg:
        if n_reg >= 35:
            break
        for sec in section_doc(clean(ex["text"])):
            if n_reg >= 35:
                break
            docs.append({"lang": "en", "source": "legal_reg", "text": sec})
            n_reg += 1
    print("reg:", n_reg)

    with open("data/legal_corpus.jsonl", "w") as f:
        for i, d in enumerate(docs):
            d["doc_id"] = i
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("TOTAL legal docs:", len(docs), "mean len:", sum(len(d["text"]) for d in docs) // len(docs))


if __name__ == "__main__":
    main()
