"""Assemble a mixed EN + PT-BR corpus for chunking distillation.

Sources (all on-disk or fast to download):
  - little-stories-en_US-pt_BR  (cached): parallel EN/PT short stories
  - opus_books en-pt            (cached): parallel EN/PT book sections
  - AllTripletsMsMarco-PTBR     (cached): PT-BR passages
  - dominguesm/wikipedia-ptbr   (download): PT wiki articles
  - cnn_dailymail               (download): EN news articles
  - legalbench consumer contracts (download, reconstructed from chunk tiles)
"""
import json
import random
import re
import glob

import pandas as pd
from datasets import load_dataset

random.seed(1337)

OUT = "data/corpus.jsonl"
MIN_LEN, MAX_LEN = 350, 2500

HF = "/home/moro/.cache/huggingface/hub"


def clean(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def section(doc: str, target: int = 900, overlap: int = 0):
    """Split a long document into ~target-char sections on paragraph boundaries."""
    paras = [p for p in re.split(r"\n\s*\n", doc) if len(p.strip()) >= 40]
    out, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) > target and cur:
            out.append(clean("\n\n".join(cur)))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        out.append(clean("\n\n".join(cur)))
    return [d for d in out if MIN_LEN <= len(d) <= MAX_LEN]


def add_stories(docs):
    p = glob.glob(f"{HF}/datasets--orion-research--little-stories-en_US-pt_BR/snapshots/*/ors/notebooks/Datasets/reqs_roneneldan/final_combined_dataset.parquet")[0]
    df = pd.read_parquet(p)
    pt = df["input"].tolist()
    en = df["output"].tolist()
    for t in random.sample(en, 400):
        if MIN_LEN <= len(t) <= MAX_LEN:
            docs.append({"lang": "en", "source": "stories", "text": clean(t)})
    for t in random.sample(pt, 400):
        if MIN_LEN <= len(t) <= MAX_LEN:
            docs.append({"lang": "pt", "source": "stories", "text": clean(t)})
    print("stories:", len(docs))


def add_books(docs):
    p = glob.glob(f"{HF}/datasets--opus_books/snapshots/*/en-pt/train-00000-of-00001.parquet")[0]
    df = pd.read_parquet(p)
    n_en = n_pt = 0
    for tr in df["translation"].tolist():
        for lang, key in [("en", "en"), ("pt", "pt")]:
            if n_en >= 120 and n_pt >= 120:
                break
            if lang == "en" and n_en >= 120:
                continue
            if lang == "pt" and n_pt >= 120:
                continue
            txt = clean(tr.get(key, ""))
            for sec in section(txt):
                if n_en >= 120 and n_pt >= 120:
                    break
                if lang == "en" and n_en < 120:
                    docs.append({"lang": "en", "source": "books", "text": sec})
                    n_en += 1
                elif lang == "pt" and n_pt < 120:
                    docs.append({"lang": "pt", "source": "books", "text": sec})
                    n_pt += 1
    print("books:", len(docs))


def add_pt_triplets(docs):
    files = sorted(glob.glob(f"{HF}/datasets--cnmoro--AllTripletsMsMarco-PTBR/snapshots/*/data/train-0000*.parquet"))[:5]
    texts = []
    for f in files:
        df = pd.read_parquet(f, columns=["positive"])
        texts.extend(df["positive"].tolist())
    seen = set()
    count = 0
    for t in texts:
        t = clean(t)
        if len(t) < MIN_LEN or len(t) > MAX_LEN or t in seen:
            continue
        seen.add(t)
        docs.append({"lang": "pt", "source": "triplets", "text": t})
        count += 1
        if count >= 400:
            break
    print("triplets:", len(docs))


def add_pt_wiki(docs):
    ds = load_dataset("dominguesm/wikipedia-ptbr-20230601", split="train", streaming=True)
    count = 0
    for ex in ds:
        text = clean(ex.get("text", ""))
        title = ex.get("title", "")
        if title:
            text = f"{title}\n\n{text}"
        for sec in section(text):
            if count >= 200:
                break
            docs.append({"lang": "pt", "source": "wiki", "text": sec})
            count += 1
        if count >= 200:
            break
    print("pt_wiki:", len(docs))


def add_en_news(docs):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    paths = []
    for i in range(3):
        paths.append(hf_hub_download("cnn_dailymail", f"3.0.0/train-0000{i}-of-00003.parquet",
                                     repo_type="dataset"))
    frames = [pd.read_parquet(p)[["article"]] for p in paths]
    df = pd.concat(frames, ignore_index=True)
    count = 0
    for art in df["article"].tolist():
        text = clean(art)
        for sec in section(text):
            if count >= 200:
                break
            docs.append({"lang": "en", "source": "news", "text": sec})
            count += 1
        if count >= 200:
            break
    print("en_news:", len(docs))


def add_legal(docs):
    """Reconstruct full contracts from the paper's chunk-retrieval tiles."""
    ds = load_dataset(
        "bowang0911/LegalBenchConsumerContractsQAChunkRetrieval",
        "documents-512", split="test", streaming=True,
    )
    by_url = {}
    for ex in ds:
        by_url.setdefault(ex["source_url"], []).append(ex)
    count = 0
    for url, chunks in by_url.items():
        chunks.sort(key=lambda c: c["chunk_idx"])
        full = " ".join(c["chunk"] for c in chunks)
        full = clean(full)
        for sec in section(full):
            if count >= 200:
                break
            docs.append({"lang": "en", "source": "legal", "text": sec})
            count += 1
        if count >= 200:
            break
    print("legal:", len(docs))


def main():
    docs = []
    add_stories(docs)
    add_books(docs)
    add_pt_triplets(docs)
    add_pt_wiki(docs)
    add_en_news(docs)
    add_legal(docs)
    random.shuffle(docs)
    with open(OUT, "w") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({"doc_id": i, **d}, ensure_ascii=False) + "\n")
    langs = {}
    for d in docs:
        langs[d["lang"]] = langs.get(d["lang"], 0) + 1
    print("TOTAL:", len(docs), langs)


if __name__ == "__main__":
    main()
