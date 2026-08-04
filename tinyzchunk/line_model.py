"""Line-level boundary model (numpy inference).

Predicts, for every non-blank line in a document, whether a new chunk/unit
starts there, using a window of neighbouring line features.  Trained in torch,
runs with numpy.
"""
import numpy as np

from .features import extract_features, normalize_text


def line_starts_of(text):
    """Index of the first character of every non-blank line."""
    starts = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\n":
            i += 1
            continue
        starts.append(i)
        j = text.find("\n", i)
        i = n if j < 0 else j + 1
    return np.array(starts, dtype=np.int64)


def window_stack(feats, ctx):
    """(L, F) per-line features -> (L, (2*ctx+1)*F) with neighbours concatenated.

    Out-of-range neighbours are zero-padded, matching training.
    """
    L, F = feats.shape
    W = 2 * ctx + 1
    out = np.zeros((L, W * F), dtype=np.float32)
    for r in range(-ctx, ctx + 1):
        col = (r + ctx) * F
        # neighbour r of line i is line i+r; clamp so that a document with
        # fewer lines than the window simply leaves the far slots zeroed
        lo, hi = max(0, -r), min(L, L - r)
        if lo >= hi:
            continue
        out[lo:hi, col:col + F] = feats[lo + r:hi + r]
    return out


class LineBoundaryModel:
    def __init__(self, weights: dict):
        self.Ws = []
        i = 1
        while f"W{i}" in weights:
            self.Ws.append((
                np.ascontiguousarray(weights[f"W{i}"].T, dtype=np.float32),
                np.asarray(weights[f"b{i}"], dtype=np.float32)))
            i += 1
        self.ctx = int(np.ravel(weights.get("CTX", np.array([2])))[0])

    @classmethod
    def from_npz(cls, path):
        with np.load(path) as w:
            return cls({k: w[k] for k in w.files})

    @staticmethod
    def line_starts(text):
        return line_starts_of(text)

    def score_lines(self, text, X=None, starts=None):
        """Per-line boundary scores in [0, 1] (aligned to line_starts).

        Pass a precomputed feature matrix `X` (and its line `starts`) to avoid
        re-extracting features that the caller already has.
        """
        if X is None:
            text = normalize_text(text)
            starts = line_starts_of(text)
            X = extract_features(text, normalize=False)
        if len(starts) == 0:
            return np.zeros(0)
        h = window_stack(X[starts], self.ctx)
        for W, b in self.Ws[:-1]:
            h = np.maximum(0.0, h @ W + b)
        logits = h @ self.Ws[-1][0] + self.Ws[-1][1]
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).reshape(-1)

    def boundaries(self, text, threshold=0.5):
        scores = self.score_lines(text)
        starts = line_starts_of(normalize_text(text))
        return [int(p) for p, s in zip(starts, scores) if s >= threshold]


if __name__ == "__main__":
    m = LineBoundaryModel.from_npz("tinyzchunk/line_weights.npz")
    t = ("Qual o nome do espaço? Resposta aqui.\n"
         "Quem é o responsável? Maria Silva\nOnde fica? Asa 2\n")
    print("scores:", [round(x, 2) for x in m.score_lines(t)])
