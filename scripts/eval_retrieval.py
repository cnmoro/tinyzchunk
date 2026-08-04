"""End-to-end evaluation on the zChunk paper's own chunk-retrieval benchmark.

Uses bowang0911/LegalBenchConsumerContractsQAChunkRetrieval: full consumer
contracts with gold answer fragments.  For each query fragment we measure, under
each chunking strategy, whether the gold fragment stays inside a single chunk
(not split) and how much surrounding noise it carries (chunk size / fragment
size).  This mirrors the paper's signal-to-noise motivation.
"""
import argparse
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from tinyzchunk.chunker import Chunker


def fixed_chunks(text, size=512):
    return [text[i:i + size] for i in range(0, len(text), size)]


def sentence_chunks(text, target=500):
    if len(text) <= target:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        j = i + target
        seg = text[i:j]
        ends = [m.end() for m in re.finditer(r"[.!?;:]", seg)]
        if ends:
            j = i + max(ends)
        chunks.append(text[i:j])
        i = j
    return [c for c in chunks if c.strip()]


def paragraph_chunks(text):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, curlen = [], [], 0
    for p in paras:
        if curlen + len(p) > 800 and cur:
            chunks.append("\n\n".join(cur))
            cur, curlen = [], 0
        cur.append(p)
        curlen += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="tinyzchunk/weights.npz")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("bowang0911/LegalBenchConsumerContractsQAChunkRetrieval",
                      "documents-512", split="test", streaming=True)
    docs = {}
    for ex in ds:
        docs.setdefault(ex["source_url"], []).append(ex)
    full = {}
    for u, ch in docs.items():
        ch = sorted(ch, key=lambda c: c["chunk_idx"])
        full[u] = "".join(c["chunk"] for c in ch)

    queries = list(load_dataset(
        "bowang0911/LegalBenchConsumerContractsQAChunkRetrieval",
        "queries", split="test", streaming=True))

    # group fragments by source doc
    frag_by_doc = {}
    for ex in queries:
        urls = ex["source_url"]
        if isinstance(urls, list):
            for u, s, e in zip(urls, ex["frag_start_char"], ex["frag_end_char"]):
                frag_by_doc.setdefault(u, []).append((s, e))
        else:
            frag_by_doc.setdefault(urls, []).extend(
                zip(ex["frag_start_char"], ex["frag_end_char"]))

    c = Chunker(weights_path=args.weights)
    methods = {
        "student": lambda t: c.chunk(t),
        "fixed512": lambda t: fixed_chunks(t),
        "sentence500": lambda t: sentence_chunks(t),
        "paragraph": lambda t: paragraph_chunks(t),
    }

    print(f"{'method':<14} {'frags':>5} {'split%':>7} {'noise50':>8} {'noise90':>8}")
    for name, chunkfn in methods.items():
        frags = 0
        split_total = 0
        noises = []
        for u, frags_list in frag_by_doc.items():
            doc = full.get(u)
            if doc is None:
                continue
            chunks = chunkfn(doc)
            c_start, c_end = [], []
            pos = 0
            for ch in chunks:
                s = doc.find(ch, pos)
                if s < 0:
                    s = pos
                c_start.append(s)
                c_end.append(s + len(ch))
                pos = s + len(ch)
            for (fs, fe) in frags_list:
                frags += 1
                contained = [i for i in range(len(chunks))
                             if c_start[i] <= fs and c_end[i] >= fe]
                if not contained:
                    split_total += 1
                else:
                    ci = contained[0]
                    noises.append((c_end[ci] - c_start[ci]) / max(fe - fs, 1))
        noises = np.array(noises)
        print(f"{name:<14} {frags:>5} {100*split_total/frags:>6.1f}% "
              f"{np.percentile(noises,50):>8.2f} {np.percentile(noises,90):>8.2f}")


if __name__ == "__main__":
    main()
