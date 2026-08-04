"""Build a diverse held-out eval corpus from real HuggingFace datasets."""
import json, re
from datasets import load_dataset

def clean(t):
    return re.sub(r"[ \t]+", " ", str(t)).replace("\r", "").strip()

docs = []
# 1. CDC COVID FAQ -> Q&A Q&A script docs
n=0
faq = load_dataset("CShorten/CDC-COVID-FAQ", split="train", streaming=True)
for ex in faq:
    if n >= 20: break
    q, a = clean(ex["question"]), clean(ex["answer"])
    if len(q) < 20 or len(a) < 120: continue
    docs.append({"lang":"en","source":"faq","text":f"{q} {a}"})
    n+=1
print("faq:", len(docs))

# 2. PT wikipedia articles (sectioned)
n=0
wiki = load_dataset("dominguesm/wikipedia-ptbr-20230601", split="train", streaming=True)
for ex in wiki:
    if n >= 15: break
    t = clean(ex.get("text",""))
    title = ex.get("title","")
    if not t or len(t) < 600: continue
    if title: t = title + "\n\n" + t
    docs.append({"lang":"pt","source":"wiki","text":t[:3500]})
    n+=1
print("wiki:", sum(1 for d in docs if d['source']=='wiki'))

# 3. news from existing corpus
n=0
for line in open("data/corpus.jsonl"):
    if n >= 15: break
    ex = json.loads(line)
    if ex.get("source") == "news":
        docs.append({"lang":"en","source":"news","text":ex["text"][:3000]})
        n+=1
print("news:", sum(1 for d in docs if d['source']=='news'))

# 4. legal (already built)
legal = [json.loads(l) for l in open("data/legal_corpus.jsonl")]
docs.extend(legal[:30])
print("legal:", 30)

with open("data/hf_eval_corpus.jsonl","w") as f:
    for i,d in enumerate(docs):
        d["doc_id"]=i
        f.write(json.dumps(d,ensure_ascii=False)+"\n")
import collections
print("TOTAL:", len(docs), dict(collections.Counter(d['source'] for d in docs)))
