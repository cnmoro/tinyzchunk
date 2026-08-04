"""zChunk teacher: replicates the logprob-based split-token trick.

Instead of generating output tokens, we run one forward pass over the full
prompt and read, at every position in the document, the probability the model
assigns to emitting a split token (段 = big split, 顿 = small split) as the next
token.  Those probabilities ARE the chunk-boundary signal.

Produces, per document, a JSONL record with per-character teacher scores:
  big_logp   : log P(next = 段)        (big/section split signal)
  small_logp : log P(next = 顿)        (small/sentence split signal)
  cont_logp  : log P(actual next token) (continuation confidence)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BIG = "\u6bb5"    # 段
SMALL = "\u987f"  # 顿

SYSTEM = """Your job is to act as a "Chunker", for use in RAG pipelines. The user will provide a long document.

You, the assistant, should repeat the exact same message verbatim. EXCEPT, you should insert split tokens throughout the passage.

# Instructions

- For big splits, please use `{BIG}` as the "big split token" separator.
- For small splits, please use `{SMALL}` as the "small split token" separator.
- For example, in text document, small splits will be per-sentence, and big splits will be per-section. Do the big split BEFORE the header that defined the section.
- You may get a user message that is unstructured or not structured cleanly. Still try to split that input as best as you can, even if it just means doing a small split every 100 characters, and a big split every 500 characters.
- You should prefer to wait until the end of a newline or period to break, instead of breaking one or two tokens before that. Of course, if there are no newlines or periods, pick some other reasonable breakpoints instead.
- Please note that you will sometimes not see your own splits in your previous output, that's ok, you MUST continue to try to output split tokens""".format(BIG=BIG, SMALL=SMALL)

EXAMPLE_INPUT = """VI Polices and Terms

1. INTELLECTUAL PROPERTY COMPLAINTS
Amazon respects the intellectual property of others. If you believe that your intellectual property rights are being infringed, please follow our Notice and Procedure for Making Claims of Copyright Infringement.

2. RISK OF LOSS
All purchases of physical items from Amazon are made pursuant to a shipment contract. This means that the risk of loss and title for such items pass to you upon our delivery to the carrier.

3. RETURNS, REFUNDS AND TITLE
Amazon does not take title to returned items until the item arrives at our fulfillment center. At our discretion, a refund may be issued without requiring a return. In this situation, Amazon does not take title to the refunded item. For more information about our returns and refunds, please see our Returns Center."""

EXAMPLE_OUTPUT = f"""{BIG}VI Polices and Terms

{BIG}1. INTELLECTUAL PROPERTY COMPLAINTS
{SMALL}Amazon respects the intellectual property of others.{SMALL} If you believe that your intellectual property rights are being infringed, please follow our Notice and Procedure for Making Claims of Copyright Infringement.

{BIG}2. RISK OF LOSS
{SMALL}All purchases of physical items from Amazon are made pursuant to a shipment contract.{SMALL} This means that the risk of loss and title for such items pass to you upon our delivery to the carrier.

{BIG}3. RETURNS, REFUNDS AND TITLE
{SMALL}Amazon does not take title to returned items until the item arrives at our fulfillment center.{SMALL} At our discretion, a refund may be issued without requiring a return.{SMALL} In this situation, Amazon does not take title to the refunded item.{SMALL} For more information about our returns and refunds, please see our Returns Center."""

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SECTION = 1800
OVERLAP = 200


class Teacher:
    def __init__(self, model_name=MODEL, device="cuda"):
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=q, device_map="cuda",
            torch_dtype=torch.float16, use_cache=False)
        self.model.eval()
        self.big_id = self.tok.encode(BIG, add_special_tokens=False)[0]
        self.small_id = self.tok.encode(SMALL, add_special_tokens=False)[0]
        assert self.big_id is not None and self.small_id is not None
        self.prefix_ids, self.prefix_len = self._build_prefix()

    def _build_prefix(self):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": EXAMPLE_INPUT},
            {"role": "assistant", "content": EXAMPLE_OUTPUT},
            {"role": "user", "content": "{DOC}"},
        ]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        head, tail = s.split("{DOC}")
        self._tail_ids = self.tok.encode(tail, add_special_tokens=False)
        ids = self.tok.encode(head, add_special_tokens=False)
        return ids, len(ids)

    @torch.no_grad()
    def score_section(self, doc_tokens):
        """doc_tokens: list of token ids for one document section.

        For each doc token j, returns:
          big_logp[j]   = P(next = 段  | context through token j)  (split AFTER token j)
          small_logp[j] = P(next = 顿  | context through token j)
          cont_logp[j]  = P(token j   | context through token j-1) (surprisal)
        """
        n = len(doc_tokens)
        ids = self.prefix_ids + doc_tokens + self._tail_ids
        input_ids = torch.tensor([ids], device="cuda")
        logits = self.model(input_ids).logits[0].float()  # (T, V)
        lse = logits.logsumexp(-1)  # (T,)
        doc_start = self.prefix_len
        # split signal: prediction AFTER each doc token (row doc_start+j)
        sel = logits[doc_start: doc_start + n][:, [self.big_id, self.small_id]]
        split_logp = sel - lse[doc_start: doc_start + n].unsqueeze(1)  # (n, 2)
        # surprisal: row doc_start-1+j predicts token j
        cont_ids = torch.tensor(doc_tokens, device="cuda")
        cont_logp = logits[doc_start - 1: doc_start + n - 1].gather(
            1, cont_ids.unsqueeze(1)).squeeze(1) - lse[doc_start - 1: doc_start + n - 1]
        big = split_logp[:, 0].detach().cpu().numpy()
        small = split_logp[:, 1].detach().cpu().numpy()
        cont = cont_logp.detach().cpu().numpy()
        return big, small, cont

    def score(self, text):
        """Return per-character arrays of big/small/cont logprobs for `text`.

        big[i] = teacher's log-prob of a BIG split occurring at char position i
                 (i.e. between char i-1 and i).  The chunk boundary signal.
        """
        n_chars = len(text)
        big = np.full(n_chars, -100.0)
        small = np.full(n_chars, -100.0)
        cont = np.full(n_chars, -100.0)

        doc_ids = self.tok.encode(text, add_special_tokens=False)
        if os.environ.get("TZ_DEBUG"):
            print(f"[dbg] doc_ids={len(doc_ids)}", file=sys.stderr, flush=True)
        # (start, end) char offsets of each token in the original text
        toks = []
        running = 0
        for tid in doc_ids:
            tok = self.tok.decode([tid])
            toks.append((running, running + len(tok)))
            running += len(tok)
        # pad so char indices cover full text (tokenizer may drop/normalize whitespace)
        for i in range(n_chars):
            pass  # fill-forward below handles gaps

        # sectioned inference with overlap (mirrors zchunk main_query)
        TOK_SECTION = SECTION // 3
        TOK_OVERLAP = OVERLAP // 3
        i = 0
        while i < len(doc_ids):
            start = i
            end = min(i + TOK_SECTION, len(doc_ids))
            t0 = time.time()
            b, s, c = self.score_section(doc_ids[start:end])
            if os.environ.get("TZ_DEBUG"):
                print(f"[dbg] section {start}:{end} {time.time()-t0:.2f}s", file=sys.stderr, flush=True)
            for j in range(len(b)):
                end_char = min(toks[start + j][1], n_chars - 1)
                big[end_char] = b[j]
                small[end_char] = s[j]
                cont[end_char] = c[j]
            if end >= len(doc_ids):
                break
            i = end - TOK_OVERLAP

        # fill forward so every char has a value (token boundary -> end of token)
        for arr in (big, small, cont):
            last = -100.0
            for i in range(n_chars):
                if arr[i] != -100.0:
                    last = arr[i]
                else:
                    arr[i] = last
        return big, small, cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="data/labels/labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    teacher = Teacher()
    docs = [json.loads(l) for l in open(args.corpus)]
    if args.limit:
        docs = docs[: args.limit]

    f = open(args.out, "w")
    t0 = time.time()
    for k, d in enumerate(docs):
        text = d["text"]
        t_doc = time.time()
        try:
            big, small, cont = teacher.score(text)
        except torch.cuda.OutOfMemoryError:
            print("OOM on", d["doc_id"], "skipping", file=sys.stderr)
            continue
        dt_doc = time.time() - t_doc
        rec = {
            "doc_id": d["doc_id"], "lang": d["lang"], "source": d["source"],
            "text": text,
            "big_logp": big.tolist(), "small_logp": small.tolist(),
            "cont_logp": cont.tolist(),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if (k + 1) % 25 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"[{k+1}/{len(docs)}] {rate:.2f} docs/s  ETA {(len(docs)-k-1)/rate/60:.1f}m  last_doc={dt_doc:.1f}s", file=sys.stderr)
        elif k < 40 and dt_doc > 5:
            print(f"SLOW doc {k}: {dt_doc:.1f}s len={len(text)}", file=sys.stderr)
    f.close()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
