"""Train the line-level boundary model.

Operates per LINE (with a context window of neighbouring lines), which is far
more reliable than per-character detection for line-structured documents
(Q&A scripts, schedules, bios, rosters, sectioned prose).  Still a tiny MLP with
numpy inference.

Per-line features are kept as a compact (n_lines, F) matrix and the context
window is gathered on the GPU per batch, so the corpus can grow far beyond what
a materialised (n_lines, W*F) matrix would allow.
"""
import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from tinyzchunk.features import extract_features, feature_schema_hash, n_features
from tinyzchunk.labels import normalized_record
from tinyzchunk.line_model import line_starts_of

CTX = 5  # context lines each side
HIDDEN = (384, 192, 64)

DEFAULT_LABELS = [
    "data/labels/labels.jsonl",
    "data/struct_labels/labels.jsonl",
    "data/noisy_train/labels.jsonl",
    "data/canarim_labels/labels.jsonl",
    "data/synth_labels/labels.jsonl",
    "data/aug_labels/labels.jsonl",
]


def build_lines(records, ctx=CTX):
    """-> feats (N, F) float32, y (N,) float32, nbr (N, 2*ctx+1) int64 (-1 = pad)."""
    feats, ys, nbrs = [], [], []
    base = 0
    W = 2 * ctx + 1
    for rec in records:
        text, yb = normalized_record(rec, "big")
        if len(text) < 8 or "\n" not in text:
            continue
        starts = line_starts_of(text)
        if len(starts) < 2:
            continue
        X = extract_features(text, normalize=False)
        f = X[starts]
        L = len(starts)
        y = np.zeros(L, dtype=np.float32)
        # a line is positive when a labelled boundary falls on (or just before)
        # its first character, which absorbs the newline offset of the teacher
        for si, st in enumerate(starts):
            lo = max(0, st - 2)
            if yb[lo:st + 1].max() > 0:
                y[si] = 1.0
        idx = np.arange(L)
        nbr = idx[:, None] + np.arange(-ctx, ctx + 1)[None, :]
        nbr = np.where((nbr >= 0) & (nbr < L), nbr + base, -1)
        feats.append(f.astype(np.float32))
        ys.append(y)
        nbrs.append(nbr.astype(np.int64))
        base += L
    if not feats:
        raise SystemExit("no usable training records")
    return (np.concatenate(feats), np.concatenate(ys),
            np.concatenate(nbrs).reshape(-1, W))


class Net(nn.Module):
    def __init__(self, F, ctx=CTX, hidden=HIDDEN):
        super().__init__()
        dims = [F * (2 * ctx + 1), *hidden, 1]
        self.fcs = nn.ModuleList(nn.Linear(dims[i], dims[i + 1])
                                 for i in range(len(dims) - 1))

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = torch.relu(fc(x))
        return self.fcs[-1](x).squeeze(-1)


def gather(feats_pad, nbr_batch):
    """nbr uses -1 for padding; row 0 of feats_pad is a zero row."""
    B, W = nbr_batch.shape
    return feats_pad[nbr_batch + 1].reshape(B, -1)


def f1_at(pred_scores, y, t):
    pred = pred_scores >= t
    tp = float((pred & (y > 0.5)).sum())
    if pred.sum() == 0 or y.sum() == 0:
        return 0.0, 0.0, 0.0
    pr = tp / float(pred.sum())
    rc = tp / float(y.sum())
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--out", default="tinyzchunk/line_weights.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=12.0)
    ap.add_argument("--val-frac", type=float, default=0.08)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(0)

    records = []
    for p in args.labels:
        try:
            recs = [json.loads(l) for l in open(p)]
        except FileNotFoundError:
            print(f"  (skipping missing {p})")
            continue
        print(f"  {p}: {len(recs)} docs")
        records.extend(recs)
    random.shuffle(records)
    n_val = max(60, int(len(records) * args.val_frac))
    val_records, train_records = records[:n_val], records[n_val:]

    t0 = time.time()
    Xf, y, nbr = build_lines(train_records)
    Xvf, yv, nbrv = build_lines(val_records)
    F = Xf.shape[1]
    assert F == n_features(), (F, n_features())
    print(f"train lines {len(y)} ({100*y.mean():.2f}% positive), "
          f"val lines {len(yv)} ({100*yv.mean():.2f}%), F={F}, "
          f"featurized in {time.time()-t0:.0f}s")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zero = np.zeros((1, F), dtype=np.float32)
    Xp = torch.from_numpy(np.concatenate([zero, Xf])).to(dev)
    Xvp = torch.from_numpy(np.concatenate([zero, Xvf])).to(dev)
    nbr_t = torch.from_numpy(nbr).to(dev)
    nbrv_t = torch.from_numpy(nbrv).to(dev)
    yt = torch.from_numpy(y).to(dev)
    yv_t = torch.from_numpy(yv).to(dev)

    net = Net(F).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pw = torch.tensor(args.pos_weight, device=dev)
    n = len(y)

    best = (-1.0, None, 0.5)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        total = 0.0
        t0 = time.time()
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            out = net(gather(Xp, nbr_t[idx]))
            loss = nn.functional.binary_cross_entropy_with_logits(
                out, yt[idx], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(yv), 16384):
                pv.append(torch.sigmoid(net(gather(Xvp, nbrv_t[i:i + 16384]))))
            pv = torch.cat(pv)
        bf1, bt = 0.0, 0.5
        for t in np.arange(0.20, 0.91, 0.05):
            f1, _, _ = f1_at(pv, yv_t, float(t))
            if f1 > bf1:
                bf1, bt = f1, float(t)
        if bf1 > best[0]:
            best = (bf1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, bt)
        print(f"epoch {ep:>2}: bce={total/n:.4f} val_f1={bf1:.4f}@{bt:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    net.load_state_dict(best[1])
    print(f"BEST val line-F1 {best[0]:.4f} at threshold {best[2]:.2f}")

    from tinyzchunk.model import export_weights
    export_weights(net, args.out, extra={"CTX": np.array([CTX]),
                                         "BEST_T": np.array([best[2]])})
    print("exported:", args.out, "schema:", feature_schema_hash())


if __name__ == "__main__":
    main()
