"""tinyzchunk: a GPU-free, distilled chunker for RAG pipelines.

Distilled from an LLM teacher into two small MLPs that run with numpy alone.
Character-level features only, so it works for English, Brazilian Portuguese and
other latin-script languages without a tokenizer.

The pipeline is:

    normalize (length preserving)
      -> line-level unit scores (primary) blended with char-level scores
      -> veto boundaries inside code fences / tables
      -> snap off mid-word cuts
      -> merge continuations and undersized chunks (removes boundaries)
      -> enforce max_chunk_chars (adds boundaries)
      -> slice the ORIGINAL text

Normalization maps one character to one character (and deletes only the CR of a
CRLF pair, which is tracked by an index map), so every boundary index is valid in
the caller's original text and chunks are exact substrings of the input.
"""
import os
import re

import numpy as np

from .features import (extract_features, feature_schema_hash, n_features,
                       normalize_text, normalize_with_map)
from .line_model import LineBoundaryModel
from .model import TinyChunkModel

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_DIR, "weights.npz")       # bundled in local builds
DEFAULT_LINE_WEIGHTS = os.path.join(_DIR, "line_weights.npz")
DEFAULT_REPO = "cnmoro/tinyzchunk"

# documents longer than this are scored in overlapping windows so that memory
# stays bounded (feature extraction costs ~100 float32 per character)
WINDOW_CHARS = 200_000
WINDOW_OVERLAP = 4_000

_HEADERISH_RE = re.compile(
    r"^\s*(#{1,6}\s|\*{1,2}\w|\d+[.)]\s|[A-ZÀ-Ý][A-ZÀ-Ý0-9 .&\-]{3,}\s*$)")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$")


def _hf_download(repo_id, filename):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "tinyzchunk weights are not bundled with this package. Install "
            "huggingface_hub (pip install huggingface_hub) so they can be "
            "fetched from HuggingFace.")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _protected_spans(text):
    """Char ranges that must never be split: fenced code blocks and tables."""
    spans = []
    pos = 0
    fence_start = None
    table_start = None
    for line in text.splitlines(keepends=True):
        end = pos + len(line)
        if _FENCE_RE.match(line):
            if fence_start is None:
                fence_start = pos
            else:
                spans.append((fence_start, end))
                fence_start = None
        elif fence_start is None:
            if _TABLE_RE.match(line):
                if table_start is None:
                    table_start = pos
            elif table_start is not None:
                spans.append((table_start, pos))
                table_start = None
        pos = end
    if fence_start is not None:
        spans.append((fence_start, pos))
    if table_start is not None:
        spans.append((table_start, pos))
    # a span of a single table row is not worth protecting
    return [(a, b) for a, b in spans if b - a > 1]


class Chunker:
    """Chunk text into semantically-coherent pieces without a GPU.

    Uses two distilled models:
      - a line-level model (primary): detects where a new unit starts at the
        start of a line (Q&A pairs, headers, list entries, sections);
      - a character-level model: sentence/paragraph structure, used to blend
        with the line score, as a fallback when the document has no line
        structure, and to re-split chunks that exceed max_chunk_chars.

    Weights are not shipped in the pip package; if they are not present locally
    they are fetched from the HuggingFace repo on first use (cached).

    Parameters:
      big_threshold      - sensitivity of the line-level unit detector
      small_threshold    - sensitivity of the char-level fallback
      max_chunk_chars    - hard ceiling; longer chunks are re-split
      min_chunk_chars    - chunks shorter than this are merged away
      char_blend         - weight of the char model in the line-start score
      adaptive           - retry with a lower threshold when a structured
                           document yields no boundaries at all
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS,
                 line_weights_path=DEFAULT_LINE_WEIGHTS,
                 repo_id=DEFAULT_REPO,
                 big_threshold=0.50, small_threshold=0.50,
                 max_chunk_chars=2500, min_chunk_chars=100,
                 char_blend=0.15, adaptive=True, nms=4):
        if weights_path is None or not os.path.exists(weights_path):
            weights_path = _hf_download(repo_id, "weights.npz")
        if line_weights_path is None or not os.path.exists(line_weights_path):
            line_weights_path = _hf_download(repo_id, "line_weights.npz")
        self.char_model = TinyChunkModel.from_npz(weights_path)
        self.line_model = (LineBoundaryModel.from_npz(line_weights_path)
                           if os.path.exists(line_weights_path) else None)
        self._check_schema(weights_path, line_weights_path)
        self.big_threshold = big_threshold
        self.small_threshold = small_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.char_blend = char_blend
        self.adaptive = adaptive
        self.nms = nms

    def _check_schema(self, *paths):
        """Fail loudly when weights were trained against a different feature set."""
        want = feature_schema_hash()
        n_feat = n_features()
        for p in paths:
            if not p or not os.path.exists(p):
                continue
            with np.load(p) as w:
                got = str(w["SCHEMA"]) if "SCHEMA" in w.files else None
                in_dim = int(w["W1"].shape[1]) if "W1" in w.files else None
            if got is not None and got != want:
                raise ValueError(
                    f"weights at {p} were trained with feature schema {got}, but "
                    f"this tinyzchunk build produces {want}. Upgrade tinyzchunk or "
                    "re-download the matching weights.")
            if in_dim is not None and in_dim % n_feat != 0:
                raise ValueError(
                    f"weights at {p} expect {in_dim} inputs, which is not a multiple "
                    f"of the {n_feat} features this build produces. Upgrade "
                    "tinyzchunk or re-download the matching weights.")

    # ------------------------------------------------------------------ scoring

    def score(self, text: str):
        """Return (big, small) per-character split scores in [0, 1]."""
        X = extract_features(text)
        if len(X) == 0:
            return np.zeros(0), np.zeros(0)
        return self.char_model.score(X)

    def line_boundaries(self, text: str, threshold=None):
        """Line-start boundaries from the line-level model."""
        if self.line_model is None:
            return []
        t = self.big_threshold if threshold is None else threshold
        return self.line_model.boundaries(text, t)

    def _scored_line_starts(self, text, X=None, big=None):
        """(starts, blended_scores) for every non-blank line."""
        if self.line_model is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        starts = self.line_model.line_starts(text)
        if len(starts) == 0:
            return starts, np.zeros(0)
        if X is None:
            X = extract_features(text, normalize=False)
        scores = self.line_model.score_lines(text, X=X, starts=starts)
        if self.char_blend > 0:
            if big is None:
                big, _ = self.char_model.score(X)
            if len(big):
                cs = big[np.clip(starts, 0, len(big) - 1)]
                scores = (1.0 - self.char_blend) * scores + self.char_blend * cs
        return starts, scores

    # --------------------------------------------------------------- boundaries

    def boundaries(self, text: str, window=3, nms=6):
        """Return list of (char_index, kind) boundaries, indexed into `text`."""
        if not text:
            return []
        norm, index_map = normalize_with_map(text)
        bds = (self._windowed_boundaries(norm) if len(norm) > WINDOW_CHARS
               else self._boundaries_norm(norm, window))
        if index_map is None:
            return bds
        return [(int(index_map[p]) if p < len(index_map) else len(text), k)
                for p, k in bds]

    def _windowed_boundaries(self, text):
        """Score very long documents in windows aligned to line breaks.

        Windows are cut at a newline and each seam is itself a boundary, so a
        chunk never straddles two windows and memory stays bounded.
        """
        n = len(text)
        out = []
        start = 0
        while start < n:
            end = min(n, start + WINDOW_CHARS)
            if end < n:  # cut the window at a line break
                nl = text.rfind("\n", start + WINDOW_CHARS // 2, end)
                if nl > start:
                    end = nl + 1
            for p, kind in self._boundaries_norm(text[start:end], 3):
                q = start + p
                if not out or q - out[-1][0] > 8:
                    out.append((q, kind))
            if end >= n:
                break
            if not out or end - out[-1][0] > 8:
                out.append((end, "big"))
            start = end
        return out

    def _boundaries_norm(self, text, window=3):
        n = len(text)
        # features are extracted ONCE and shared by both models
        X = extract_features(text, normalize=False)
        if len(X) == 0:
            return []
        big, small = self.char_model.score(X)
        small_idx = self._local_maxima(small, self.small_threshold, window)

        starts, scores = self._scored_line_starts(text, X=X, big=big)
        cand = [int(p) for p, s in zip(starts, scores) if s >= self.big_threshold]

        if not cand and self.adaptive and len(starts) > 3:
            # a structured document where nothing cleared the bar: relax until
            # something sensible appears rather than falling straight through to
            # sentence-level chunking
            for t in (0.5, 0.35, 0.25):
                cand = [int(p) for p, s in zip(starts, scores) if s >= t]
                if cand:
                    break

        if not cand and n > self.max_chunk_chars:
            # no line structure at all (e.g. one very long line): fall back to
            # char-level sentence/paragraph scores.  A document that already
            # fits inside max_chunk_chars is left whole rather than being
            # chopped into sentences.
            cand = [int(p) for p in self._local_maxima(big, self.big_threshold, window)]

        protected = _protected_spans(text)
        cand = [p for p in cand if not any(a < p < b for a, b in protected)]
        cand = sorted({self._snap(p, text) for p in cand if 0 < p < n})

        # ---- merge pass: drop boundaries (continuations, undersized chunks) ----
        cand = self._merge_boundaries(cand, text)

        # ---- split pass: enforce max_chunk_chars ----
        final = []
        prev = 0
        for pos in cand + [n]:
            while pos - prev > self.max_chunk_chars:
                target = prev + self.max_chunk_chars
                mids = [i for i in small_idx
                        if prev + self.min_chunk_chars < i < pos
                        and abs(i - target) <= self.max_chunk_chars // 2
                        and not any(a < i < b for a, b in protected)]
                if mids:
                    best = int(max(mids, key=lambda i: small[i]))
                    kind = "small"
                else:
                    best = self._snap_hard(target, text, protected)
                    kind = "hard"
                if best <= prev:  # no progress possible; give up on this span
                    break
                final.append((best, kind))
                prev = best
            if pos < n:
                final.append((pos, "big"))
            prev = pos
        return final

    def _merge_boundaries(self, cand, text):
        """Remove boundaries that would produce fragments or undersized chunks.

        A chunk whose first character is lowercase is a continuation of the
        previous one, so its opening boundary is dropped.  An undersized chunk
        is merged backwards, except when it looks like a heading, in which case
        it is merged *forwards* so the heading stays attached to its body.
        """
        n = len(text)
        bounds = list(cand)
        for _ in range(4):
            changed = False
            edges = [0] + bounds + [n]
            drop = set()
            for i in range(len(edges) - 1):
                s, e = edges[i], edges[i + 1]
                if i > 0 and (i - 1) in drop:
                    continue
                piece = text[s:e].strip()
                if not piece:
                    if i > 0:
                        drop.add(i - 1)
                        changed = True
                    continue
                is_cont = piece[0].islower() and len(piece) > 2
                is_tiny = len(piece) < self.min_chunk_chars
                if not (is_cont or is_tiny):
                    continue
                if is_tiny and not is_cont and self._headerish(piece) and i + 1 < len(edges) - 1:
                    drop.add(i)        # merge this heading into the NEXT chunk
                elif i > 0:
                    drop.add(i - 1)    # merge into the previous chunk
                else:
                    drop.add(i)        # first chunk: merge into the next
                changed = True
            if not changed:
                break
            bounds = [b for j, b in enumerate(bounds) if j not in drop]
        return bounds

    @staticmethod
    def _headerish(piece):
        first = piece.splitlines()[0].strip() if piece else ""
        if not first or len(first) > 90:
            return False
        if _HEADERISH_RE.match(first):
            return True
        return (first.endswith(":") or not re.search(r"[.!?]$", first)) and \
            len(first.split()) <= 12 and first[:1].isupper()

    @staticmethod
    def _snap_hard(target, text, protected=()):
        """Move a hard split to a nearby sentence end / word boundary."""
        n = len(text)

        def ok(c):
            return 0 < c < n and not any(a < c < b for a, b in protected)

        for r in range(1, 80):
            for cand in (target - r, target + r):
                if ok(cand) and text[cand - 1] in ".!?" and \
                        (text[cand] in " \n" or text[cand].isupper()):
                    return cand
        for r in range(1, 80):
            for cand in (target - r, target + r):
                if ok(cand) and text[cand] == "\n":
                    return cand
        for r in range(1, 40):
            for cand in (target - r, target + r):
                if ok(cand) and text[cand] in " \n":
                    return cand
        return target

    @classmethod
    def from_pretrained(cls, repo_id=DEFAULT_REPO, revision=None, **kwargs):
        """Load weights from a HuggingFace model repo (cached locally)."""
        if revision:
            from huggingface_hub import hf_hub_download
            char_w = hf_hub_download(repo_id=repo_id, revision=revision,
                                     filename="weights.npz")
            line_w = hf_hub_download(repo_id=repo_id, revision=revision,
                                     filename="line_weights.npz")
        else:
            char_w = _hf_download(repo_id, "weights.npz")
            line_w = _hf_download(repo_id, "line_weights.npz")
        return cls(weights_path=char_w, line_weights_path=line_w, **kwargs)

    @staticmethod
    def _local_maxima(scores, thresh, window):
        """Indices that clear `thresh` and dominate their +/- window."""
        n = len(scores)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        s = np.asarray(scores)
        pad = np.pad(s, window, mode="constant", constant_values=-np.inf)
        strided = np.lib.stride_tricks.sliding_window_view(pad, 2 * window + 1)
        is_max = s >= strided.max(axis=1)
        rising = np.ones(n, dtype=bool)
        rising[1:] = s[1:] > s[:-1]
        return np.flatnonzero((s >= thresh) & is_max & rising)

    @staticmethod
    def _snap(pos, text):
        """Move a boundary that cuts inside a word to the word start."""
        if pos <= 0 or pos >= len(text):
            return pos
        if text[pos - 1].isalnum() and text[pos].isalnum():
            q = pos - 1
            while q > 0 and text[q - 1].isalnum():
                q -= 1
            return q
        return pos

    # -------------------------------------------------------------------- chunk

    def chunk(self, text: str):
        """Return the list of chunks for `text` (exact substrings of the input)."""
        if not text or not text.strip():
            return []
        bds = [p for p, _ in self.boundaries(text)]
        cuts = [0] + [p for p in bds if 0 < p < len(text)] + [len(text)]
        chunks = []
        for s, e in zip(cuts, cuts[1:]):
            piece = text[s:e].strip()
            if piece:
                chunks.append(piece)
        return chunks or [text.strip()]


_default = None


def get_chunker(**kwargs):
    global _default
    if kwargs:
        return Chunker(**kwargs)
    if _default is None:
        _default = Chunker()
    return _default


def chunk(text: str, **kwargs):
    """One-call convenience: chunk `text` into a list of strings."""
    return get_chunker(**kwargs).chunk(text)
