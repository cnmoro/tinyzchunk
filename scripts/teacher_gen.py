"""Generation-based teacher: labels structured documents contextually.

NOTE: the prompt is deliberately few-shot. Newer instruction-tuned models obey
"reproduce the document exactly" so literally that they omit the markers
altogether -- Qwen3.5-4B emitted 1 marker per 1800 characters with a plain
instruction and 10 with this one, at identical copy fidelity.

Asks the LLM to reproduce the document inserting 【SPLIT】 markers at chunk
boundaries (explicit semantic chunking, no tag rules), then aligns the markers
back to character positions in the original text to produce per-document
boundary labels.
"""
import argparse
import difflib
import json
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MARK = "【SPLIT】"

PROMPT = """Insert the marker {M} between self-contained chunks of a document, copying everything else verbatim.

Example input:
### What is it?
A space for advice.
### Who runs it?
The support team.

Example output:
{M}### What is it?
A space for advice.
{M}### Who runs it?
The support team.

Now do the same for the document below. Insert {M} before EVERY heading and EVERY question, and wherever one self-contained topic ends and the next begins. Inserting too few markers is the main failure mode. Change nothing else: same words, same line breaks, no commentary.

Document:
{DOC}

Output:""".format(M=MARK, DOC="{DOC}")


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def load_model(model_name=DEFAULT_MODEL, device_map="cuda", quantize_head=False):
    """Load a teacher in 4-bit.

    `quantize_head` also quantizes the embedding/output head, which bitsandbytes
    normally leaves in fp16. That matters for models with a very large
    vocabulary: Qwen3.5-9B has a 248k vocab with untied embeddings, so ~2 GB of
    fp16 sits on top of the quantized layers and generation OOMs on an 8 GB
    card. Quantizing the head brings it to 6.1 GB and it runs.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           **({"llm_int8_skip_modules": []} if quantize_head else {}))
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=q,
        device_map=device_map, dtype=torch.float16).eval()
    return tok, model


def _chat(tok, prompt):
    """Render the chat prompt, disabling reasoning traces where supported."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


@torch.no_grad()
def generate_chunked(tok, model, text, max_new_tokens=1400):
    ids = tok.encode(_chat(tok, PROMPT.format(DOC=text)))
    out = model.generate(torch.tensor([ids], device="cuda"),
                         max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][len(ids):], skip_special_tokens=True)
    # reasoning models may still emit a thinking block; keep only the answer
    if "</think>" in gen:
        gen = gen.rsplit("</think>", 1)[1]
    return gen.lstrip("\n")


def align_boundaries(orig, gen_chunked):
    """Map marker positions in generated text to positions in the original."""
    parts = gen_chunked.split(MARK)
    gen_clean = "".join(parts)
    sm = difflib.SequenceMatcher(None, orig, gen_clean, autojunk=False)
    # build gen index -> orig index within 'equal' blocks
    g2o = np.full(len(gen_clean), -1, dtype=np.int64)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            g2o[j1:j2] = np.arange(i1, i2)
        elif tag == "replace":
            g2o[j1:j2] = i1 if i2 == i1 else i1  # map inserted chars to block start
    # for 'insert' (j1:j2 with no orig), map to previous equal end
    last_orig = -1
    for g in range(len(gen_clean)):
        if g2o[g] != -1:
            last_orig = g2o[g]
        else:
            g2o[g] = last_orig
    # marker positions in gen_clean
    boundaries = []
    g = 0
    for i, part in enumerate(parts):
        g += len(part)
        if i < len(parts) - 1:
            o = int(g2o[min(g, len(g2o) - 1)])
            boundaries.append(o)
    boundaries = sorted(set(b for b in boundaries if 0 <= b < len(orig)))
    return boundaries


def generate_doc_boundaries(tok, model, text, max_new_tokens, section_size=2400,
                            overlap=400, edge_margin=150):
    """Generate boundaries for a document, splitting long docs into sections."""
    if len(text) <= section_size:
        gen = generate_chunked(tok, model, text, max_new_tokens)
        if MARK not in gen:
            return None
        return align_boundaries(text, gen)

    # sectioned: windows with overlap, keep boundaries away from edges
    all_bds = []
    start = 0
    while start < len(text):
        end = min(len(text), start + section_size)
        win = text[start:end]
        gen = generate_chunked(tok, model, win, max_new_tokens)
        if MARK in gen:
            bds = align_boundaries(win, gen)
            all_bds.extend(start + b for b in bds
                           if start + edge_margin <= start + b < end - edge_margin)
        if end >= len(text):
            break
        start = end - overlap
    # dedupe near-duplicate boundaries from overlapping windows
    all_bds = sorted(set(all_bds))
    deduped = []
    for b in all_bds:
        if not deduped or b - deduped[-1] > 5:
            deduped.append(b)
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="corpus", default="data/struct_corpus.jsonl")
    ap.add_argument("--out", default="data/struct_labels/labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--quantize-head", action="store_true",
                    help="also quantize the embedding/output head (needed for\n                          large-vocabulary models on small GPUs)")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.device_map, args.quantize_head)
    docs = [json.loads(l) for l in open(args.corpus)]
    if args.limit:
        docs = docs[:args.limit]

    f = open(args.out, "w")
    t0 = time.time()
    ok = fail = 0
    for k, d in enumerate(docs):
        text = d["text"]
        try:
            bds = generate_doc_boundaries(tok, model, text, args.max_tokens)
        except torch.cuda.OutOfMemoryError:
            print("OOM", d["doc_id"], file=sys.stderr)
            continue
        if bds is None:
            fail += 1
            print(f"no markers doc {d['doc_id']}", file=sys.stderr)
            continue
        rec = {"doc_id": d["doc_id"], "lang": d["lang"], "source": d["source"],
               "text": text, "boundaries": bds}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        ok += 1
        if (k + 1) % 10 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"[{k+1}/{len(docs)}] ok={ok} fail={fail} {rate:.2f} docs/s ETA {(len(docs)-k-1)/rate/60:.1f}m", file=sys.stderr)
    f.close()
    print(f"done ok={ok} fail={fail}", file=sys.stderr)


if __name__ == "__main__":
    main()
