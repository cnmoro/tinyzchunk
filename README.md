# tinyzchunk

A **GPU-free, tokenizer-free chunker** for RAG pipelines, distilled from the
[zChunk](https://github.com/ZeroEntropy-AI/llama-chunk) idea of using an LLM's
split-token probabilities to decide where a document breaks. The teacher LLM runs
**once, offline, to produce labels**; what ships is two small MLPs that read raw
characters and run on any CPU with nothing but numpy.

Tuned for **English and Brazilian Portuguese**, and specifically hardened for the
messy text a real pipeline actually sees: PDF extractions with mid-word line
wrapping, page numbers and form feeds, OCR-mangled words, CRLF files, markdown,
code fences, tables, chat logs and legal enumerations.

```bash
pip install tinyzchunk
```

```python
from tinyzchunk import Chunker

chunker = Chunker()                    # weights fetched from HuggingFace, then cached
chunks = chunker.chunk(document)       # -> list[str]
```

```bash
python -m tinyzchunk document.txt --json
cat document.txt | python -m tinyzchunk
```

## Guarantees

Every chunk is an **exact substring** of the input, and these properties are
enforced by the test suite for any input at all:

- no content is ever dropped or duplicated;
- no chunk begins in the middle of a word;
- `max_chunk_chars` is never exceeded;
- a CRLF file chunks **identically** to the same file with unix line endings;
- fenced code blocks and markdown tables are never cut apart;
- degenerate inputs (empty, whitespace-only, one 5 MB line) do not raise.

## How it works

```
                ┌──────────────────────────┐
  corpus ─────► │ LLM teacher (GPU, once)  │──► boundary labels ──► training
                └──────────────────────────┘
                ┌──────────────────────────┐   ┌──────────────────────┐
  text ───────► │ char features (numpy)    │──►│ two tiny MLPs        │──► chunks
                └──────────────────────────┘   └──────────────────────┘
```

**Feature extraction** (`features.py`) computes 102 features per character with
no tokenizer: character classes, sentence structure, and line-level signals
broadcast across each line. Crucially the line signals are mostly *relative* —
does this line share a layout signature with its neighbours, how long is it
compared with the document average, does the previous line end mid-sentence —
so unseen formats still produce usable evidence instead of falling off a cliff.

Input is first **normalized**: unicode spaces, quotes, dashes, bullets, ellipses
and form feeds are folded to ASCII equivalents one character at a time, so
offsets stay valid. The only character ever deleted is the CR of a CRLF pair,
which is tracked with an index map.

**Two students**, both numpy-only at inference:

- `line_model.py` — the primary detector. For each line it sees a window of ±5
  neighbouring lines (11 × 102 features) and predicts whether a new unit starts
  there. This is what finds Q&A pairs, headings, schedule entries, list items and
  section starts — and what knows *not* to split wrapped prose, dense field
  lists, table rows or roster blocks.
- `model.py` — a character-level model predicting sentence/paragraph boundaries.
  It is blended into the line score, drives the fallback for text with no line
  structure, and supplies the split points when a chunk must be cut down to
  `max_chunk_chars`.

**Assembly** (`chunker.py`): line scores → veto anything inside a code fence or
table → snap off mid-word cuts → drop boundaries that would create continuation
fragments or undersized chunks (headings merge *forward*, so a heading stays with
its body) → enforce `max_chunk_chars` → slice the original text.

## Tuning

```python
Chunker(
    big_threshold=0.50,     # line-level unit detector sensitivity
    small_threshold=0.50,   # char-level sensitivity
    max_chunk_chars=2500,   # hard ceiling
    min_chunk_chars=100,    # smaller chunks are merged away
    char_blend=0.15,        # weight of the char model in the line score
    adaptive=True,          # relax the threshold rather than return nothing
)
```

`min_chunk_chars` is the strongest knob. The default of 100 biases toward
**fewer, larger chunks**: a Q&A script with 80-character turns comes back as
merged pairs rather than one chunk per line. Lower it to ~40 if you want one
chunk per structural unit.

## Evaluation

`scripts/eval_matrix.py` scores the chunker across **95 held-out scenario
buckets** — synthetic document families, boundary-preserving degradations of
them, simulated PDF/OCR noise, plus real documents — and reports boundary F1
together with the failure modes that actually hurt retrieval.

| document family | buckets | boundary F1 |
|---|---|---|
| markdown, code, tables | 12 | 0.97 |
| sectioned prose, headings, bios | 15 | 0.97 |
| legal articles and enumerations | 4 | 0.87 |
| schedules and field blocks | 15 | 0.79 |
| Q&A and FAQ | 12 | 0.78 |
| wrapped / OCR-noisy prose | 14 | 0.72 |
| lists that must *not* split | 12 | 0.70 |

**Macro F1 0.795** across all 95 buckets; **0.77** across the 36 noisy-text
buckets alone. Fragment chunks (a chunk starting mid-sentence) are **0.08%** and
oversized chunks **0%**.

Held-out real-world documents score 0.53 against a generation-teacher reference,
but that reference is itself inconsistent — some documents are labelled far more
coarsely than others — so treat it as a lower bound and read the dumped chunks.

Speed on one CPU core: ~26 ms for a 3 kB document, ~120 ms for a 21 kB one.
Weights total ≈2.1 MB.

## Reproducing the distillation

```bash
python scripts/build_synth.py       # synthetic EN+PT document families
python scripts/build_augment.py     # boundary-preserving degradations
python scripts/build_noisy.py       # simulated PDF/OCR noise

# label real prose with the LLM teachers (GPU, optional - see data/ for outputs)
python scripts/teacher.py     --in data/corpus.jsonl --out data/labels/labels.jsonl
python scripts/teacher_gen.py --in data/struct_corpus.jsonl --out data/struct_labels/labels.jsonl

python scripts/train.py             # char model  -> weights.npz
python scripts/train_line.py        # line model  -> line_weights.npz

python scripts/eval_matrix.py       # the regression matrix
pytest tests/
```

Both students train in **seconds per epoch** on a consumer GPU; the corpus is
~19k labelled documents, most of them constructed on CPU without an LLM.

### Which teacher?

`scripts/eval_teacher.py` scores a teacher against the synthetic documents,
whose boundaries are ground truth by construction — so it measures whether a
teacher is *right*, not merely whether two teachers agree (on this project two
different teachers labelling the same documents agreed at only F1 0.25).

| teacher | boundary F1 | markers vs true | speed |
|---|---|---|---|
| generation, Qwen3.5-4B | **0.779** | 1.14x | 18 s/doc |
| generation, Qwen3.5-9B | 0.768 | 1.39x | 18 s/doc |
| log-prob split-token | 0.537 | 1.38x | 0.7 s/doc |

Three results worth knowing before spending GPU hours:

- **A bigger teacher did not help.** The 9B tied the 4B on 52 of 60 documents
  and turned its extra capacity into over-splitting, not better judgement.
- **The log-prob teacher's weak aggregate is misleading.** Split by family it
  trails the generation teacher by only 0.075 on prose, and *beats* it on the
  wrapped prose that real PDF extractions consist of (0.633 vs 0.378 on
  narrow-wrapped). Use log-prob for prose and generation for structured
  documents — which is what this repo does.
- **Relabelling real prose with the better-scoring teacher changed nothing** in
  a full A/B (494 documents, corpora rebuilt, models retrained): -0.0002 on the
  teacher-independent buckets. For structured text the constructed labels are
  already perfect, so no LLM teacher is the bottleneck.

Two notes for anyone extending this, both learned the hard way:

1. The log-probability teacher places a large share of its boundaries **inside
   words**. `labels.py::snap_positions` pulls every label onto a line start or
   word start; without it the student learns to cut mid-word and every corpus
   derived from those labels inherits the damage.
2. Training-data balance is a tightrope — adding one negative pattern routinely
   breaks a positive one. Run the full matrix (`--compare` against the previous
   run) before shipping weights; never judge a change on one document family.

## Layout

```
tinyzchunk/          the library (numpy only at inference)
  chunker.py         chunk() API and assembly rules
  features.py        102 per-character features + normalization
  line_model.py      line-level unit-start model
  model.py           char-level boundary model
  labels.py          teacher labels -> clean boundary labels
scripts/             corpus builders, LLM teachers, training, evaluation
tests/               behavioural guarantees
```

Weights live in the HuggingFace repo
[`cnmoro/tinyzchunk`](https://huggingface.co/cnmoro/tinyzchunk) and are fetched
on first use; the pip package itself ships no weights.

Licence: Apache-2.0.
