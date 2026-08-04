"""Shared helpers: teacher scores / positions -> per-character boundary labels."""
import numpy as np


def detect_boundaries(x, window=3, rel_drop=3.0):
    """Find boundary positions from a fill-forwarded per-char score array.

    A boundary is the right edge of a plateau whose value is a local maximum and
    within `rel_drop` log-probs of the strongest boundary in the document
    (adaptive per-document threshold, mirrors zChunk's per-doc normalization).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.int64)
    candidates = []
    for p in range(n - 1):
        if x[p] > x[p + 1] and x[p] > -90.0:  # value drops after p -> right edge
            lo = max(0, p - window)
            hi = min(n, p + window + 1)
            if x[p] >= x[lo:hi].max():
                candidates.append(p)
    if n > 0 and x[n - 1] > -90.0 and x[n - 1] >= x[max(0, n - 1 - window):].max():
        candidates.append(n - 1)
    cand = np.array(sorted(set(candidates)), dtype=np.int64)
    if len(cand) == 0:
        return cand
    thr = x[cand].max() - rel_drop
    return cand[x[cand] >= thr]


def boundary_labels(x, window=3, rel_drop=3.0):
    """Return a 0/1 label array for every character position."""
    out = np.zeros(len(x), dtype=np.float32)
    bd = detect_boundaries(x, window, rel_drop)
    if len(bd):
        out[bd] = 1.0
    return out


def positions_to_labels(n, positions):
    """0/1 label array from explicit boundary positions."""
    out = np.zeros(n, dtype=np.float32)
    for p in positions:
        if 0 <= int(p) < n:
            out[int(p)] = 1.0
    return out


def snap_positions(text, positions, window=4):
    """Move teacher boundaries onto a legal split point.

    The log-probability teacher scores individual tokens, so a boundary can land
    in the middle of a word ("On|e day") -- a place no chunker should ever cut.
    Such a label teaches the student to fire mid-word and, worse, corrupts every
    corpus derived from it.  Each position is pulled to a nearby line start when
    there is one, otherwise back to the start of the word it landed in.
    """
    n = len(text)
    out = []
    for p in positions:
        p = int(p)
        if p <= 0 or p >= n:
            continue
        q = None
        for d in range(window + 1):
            for cand in ((p - d, p + d) if d else (p,)):
                if 0 < cand < n and text[cand - 1] == "\n" and text[cand] != "\n":
                    q = cand
                    break
            if q is not None:
                break
        if q is None:
            q = p
            if text[q - 1].isalnum() and text[q].isalnum():
                while q > 0 and text[q - 1].isalnum():
                    q -= 1
        if 0 < q < n:
            out.append(q)
    return sorted(set(out))


def normalized_record(rec, which="big"):
    """Return (normalized_text, per-char labels aligned to that text).

    Feature extraction normalizes internally, and normalization may DROP
    characters (the CR of a CRLF pair), so labels indexed against the raw text
    must be re-indexed or they silently shift.
    """
    from .features import normalize_with_map
    norm, index_map = normalize_with_map(rec["text"])
    y = labels_from_record(rec, which)
    return norm, (y if index_map is None else y[index_map])


def labels_from_record(rec, which="big", snap=True):
    """Convert a label record to per-char 0/1 boundary labels.

    Supports two record shapes:
      - logprob records: {"big_logp": [...], "small_logp": [...]}
      - explicit records: {"boundaries": [pos, ...]}
    """
    text = rec["text"]
    n = len(text)
    if "boundaries" in rec:
        pos = rec["boundaries"]
    elif which == "big" and "big_logp" in rec:
        pos = np.flatnonzero(boundary_labels(np.asarray(rec["big_logp"])))
    elif which == "small" and "small_logp" in rec:
        pos = np.flatnonzero(boundary_labels(np.asarray(rec["small_logp"])))
    else:
        return np.zeros(n, dtype=np.float32)
    if snap:
        pos = snap_positions(text, pos)
    return positions_to_labels(n, pos)
