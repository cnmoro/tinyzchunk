"""Tiny student model: a compact MLP distilled from the LLM teacher.

Trained in torch, but inference is a dependency-free numpy forward pass, so the
resulting library runs on any CPU with only numpy.
"""
import numpy as np

from .features import extract_features, feature_schema_hash, n_features

N_FEATURES = n_features()  # derived from features.py, keeps the model in sync

HIDDEN = (256, 128)


class TinyChunkModel:
    """MLP: N_FEATURES -> 256 -> 128 -> 2  (big-split score, small-split score)."""

    def __init__(self, weights: dict):
        # weights are stored transposed and C-contiguous: numpy matmul against a
        # transposed view is markedly slower than against a contiguous array
        self.layers = []
        i = 1
        while f"W{i}" in weights:
            self.layers.append((
                np.ascontiguousarray(weights[f"W{i}"].T, dtype=np.float32),
                np.asarray(weights[f"b{i}"], dtype=np.float32)))
            i += 1

    @classmethod
    def from_npz(cls, path):
        with np.load(path) as w:
            return cls({k: w[k] for k in w.files})

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, F) float32 features -> (N, 2) sigmoid scores [big, small]."""
        h = np.asarray(X, dtype=np.float32)
        for W, b in self.layers[:-1]:
            h = np.maximum(0.0, h @ W + b)
        logits = h @ self.layers[-1][0] + self.layers[-1][1]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def score(self, X: np.ndarray) -> tuple:
        s = self.forward(X)
        return s[:, 0], s[:, 1]


def torch_model(seed=0, hidden=HIDDEN, n_in=None, n_out=2):
    """Build the torch version for training (same architecture)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dims = [n_in or N_FEATURES, *hidden, n_out]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def export_weights(net, path, extra=None):
    """Copy torch Linear weights into a plain numpy dict and save as .npz.

    The feature-schema digest is stored alongside so that a mismatched
    features.py fails loudly instead of producing silent garbage.
    """
    import torch
    import torch.nn as nn

    w = {}
    with torch.no_grad():
        i = 1
        for m in net.modules():
            if isinstance(m, nn.Linear):
                w[f"W{i}"] = m.weight.detach().cpu().numpy()
                w[f"b{i}"] = m.bias.detach().cpu().numpy()
                i += 1
    w["SCHEMA"] = np.array(feature_schema_hash())
    w["F"] = np.array([N_FEATURES])
    if extra:
        w.update(extra)
    np.savez_compressed(path, **w)
    return path


if __name__ == "__main__":
    print("features:", N_FEATURES, "schema:", feature_schema_hash())
    net = torch_model()
    export_weights(net, "/tmp/dummy.npz")
    m = TinyChunkModel.from_npz("/tmp/dummy.npz")
    X = np.zeros((5, N_FEATURES), dtype=np.float32)
    print("forward ok:", m.forward(X).shape)
