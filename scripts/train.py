"""Train the character-level student model by distillation from teacher labels.

Reads label records (per-char teacher log-probs or explicit boundary lists),
builds per-char features and trains a small MLP to predict P(big boundary) and
P(small boundary) per character.  Exports weights.npz for the numpy inference
library.

Boundaries are ~0.1% of positions, so negatives are subsampled (all positives
and their neighbourhood are kept).  That keeps the training set in GPU memory
even as the corpus grows, and balances the classes for free.
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
from tinyzchunk.model import torch_model, export_weights, TinyChunkModel

DEFAULT_LABELS = [
    "data/labels/labels.jsonl",
    "data/struct_labels/labels.jsonl",
    "data/noisy_train/labels.jsonl",
    "data/canarim_labels/labels.jsonl",
    "data/synth_labels/labels.jsonl",
    "data/aug_labels/labels.jsonl",
]


def build_dataset(records, neg_frac=0.06, near=6, seed=0):
    """Features + (big, small) labels, with negatives subsampled."""
    rng = np.random.default_rng(seed)
    Xs, Bs, Ss = [], [], []
    for rec in records:
        text, yb = normalized_record(rec, "big")
        _, ys = normalized_record(rec, "small")
        if len(text) < 8:
            continue
        X = extract_features(text, normalize=False)
        pos = (yb > 0) | (ys > 0)
        # keep every positive plus a halo around it, then a sample of the rest
        halo = pos.copy()
        for k in range(1, near + 1):
            halo[k:] |= pos[:-k]
            halo[:-k] |= pos[k:]
        keep = halo | (rng.random(len(text)) < neg_frac)
        if not keep.any():
            continue
        Xs.append(X[keep])
        Bs.append(yb[keep])
        Ss.append(ys[keep])
    if not Xs:
        raise SystemExit("no usable training records")
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.stack([np.concatenate(Bs), np.concatenate(Ss)], axis=1).astype(np.float32)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--out", default="tinyzchunk/weights.npz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--max-docs", type=int, default=0)
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
    if args.max_docs:
        records = records[: args.max_docs]
    n_val = max(40, int(len(records) * args.val_frac))
    val_records, train_records = records[:n_val], records[n_val:]

    t0 = time.time()
    X, Y = build_dataset(train_records)
    Xv, Yv = build_dataset(val_records, seed=1)
    print(f"train positions {X.shape} (big {Y[:,0].sum():.0f}, small {Y[:,1].sum():.0f}), "
          f"val {Xv.shape}, featurized in {time.time()-t0:.0f}s")
    assert X.shape[1] == n_features()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pw = torch.tensor([args.pos_weight, args.pos_weight], device=dev)

    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    Xv_ = torch.from_numpy(Xv).to(dev)
    Yv_ = torch.from_numpy(Yv).to(dev)
    n = len(X)

    best = (1e9, None)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        total = 0.0
        t0 = time.time()
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(Xt[idx]), Yt[idx], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy_with_logits(
                net(Xv_), Yv_, pos_weight=pw).item()
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        print(f"epoch {ep:>2}: bce={total/n:.4f} val={vl:.4f} ({time.time()-t0:.1f}s)",
              flush=True)

    net.load_state_dict(best[1])
    print(f"BEST val bce {best[0]:.4f}")
    export_weights(net, args.out)
    print("weights exported:", args.out, "schema:", feature_schema_hash())

    student = TinyChunkModel.from_npz(args.out)
    with torch.no_grad():
        p = student.forward(Xv)
    for name, col in (("big", 0), ("small", 1)):
        gold = Yv[:, col] > 0.5
        for t in (0.5, 0.7):
            pred = p[:, col] >= t
            tp = float((pred & gold).sum())
            pr = tp / max(float(pred.sum()), 1)
            rc = tp / max(float(gold.sum()), 1)
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            print(f"  {name}@{t}: P={pr:.3f} R={rc:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
