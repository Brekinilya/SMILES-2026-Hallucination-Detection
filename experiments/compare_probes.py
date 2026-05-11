"""
Run a bunch of probing strategies on the same 5-fold split and print
accuracy / AUROC for each: baseline last token at L24, best single layer
(L14), multi-layer concat (5 layers and all 25), Mass-Mean probe at L14,
small MLP at L14, geometric features alone, geometric + L14 mix.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = np.load(REPO_ROOT / "cache" / "features.npz")
N_FOLDS = 5
SEED = 42

y_all = CACHE["train_labels"].astype(int)
N = len(y_all)
n_layers = CACHE["train_last"].shape[1]
hidden_dim = CACHE["train_last"].shape[2]
print(f"[data] N={N}  layers={n_layers}  hdim={hidden_dim}")
print(f"[data] majority baseline acc = {y_all.mean()*100:.2f}% (predict all 1)")


# ---------------------------------------------------------------------------
# Feature builders: each returns (X_train, X_test_template)
# ---------------------------------------------------------------------------
def feat_single(layer: int, pool: str) -> np.ndarray:
    return CACHE[f"train_{pool}"][:, layer, :].astype(np.float32)


def feat_multi(layers: list[int], pool: str = "last") -> np.ndarray:
    arrs = [CACHE[f"train_{pool}"][:, l, :] for l in layers]
    return np.concatenate(arrs, axis=1).astype(np.float32)


def feat_geom() -> np.ndarray:
    """Per-layer norms + inter-layer cosine + sequence length (in tokens)."""
    last = CACHE["train_last"].astype(np.float32)  # (N, L, H)
    mean = CACHE["train_mean"].astype(np.float32)  # (N, L, H)
    seq = CACHE["train_seq_lens"].astype(np.float32)

    # L2 norm of last/mean per layer  -> (N, L) each
    norms_last = np.linalg.norm(last, axis=2)
    norms_mean = np.linalg.norm(mean, axis=2)

    # cosine between successive layer hidden states (last token) -> (N, L-1)
    eps = 1e-8
    cos_last = (last[:, 1:, :] * last[:, :-1, :]).sum(axis=2) / (
        np.linalg.norm(last[:, 1:, :], axis=2) * np.linalg.norm(last[:, :-1, :], axis=2) + eps
    )
    cos_mean = (mean[:, 1:, :] * mean[:, :-1, :]).sum(axis=2) / (
        np.linalg.norm(mean[:, 1:, :], axis=2) * np.linalg.norm(mean[:, :-1, :], axis=2) + eps
    )

    feats = np.concatenate(
        [norms_last, norms_mean, cos_last, cos_mean, np.log1p(seq)[:, None]], axis=1
    )
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Mass-Mean Probe (Geometry of Truth, Marks & Tegmark 2023)
# ---------------------------------------------------------------------------
class MassMeanProbe:
    """
    Truth direction = mean(positives) - mean(negatives).
    Score = (x - midpoint) . direction
    Probability via logistic over the calibrated score.
    """

    def __init__(self, layers: list[int], pool: str = "last"):
        self.layers = layers
        self.pool = pool
        self.directions = None  # one per layer
        self.midpoints = None
        self.bias = 0.0
        self.scale = 1.0

    def _stack(self, idx: np.ndarray) -> np.ndarray:
        # (n, L_used, H)
        return CACHE[f"train_{self.pool}"][idx][:, self.layers, :].astype(np.float32)

    def fit(self, idx_train: np.ndarray, y: np.ndarray):
        H = self._stack(idx_train)            # (n, L, H)
        y_tr = y[idx_train]
        pos = H[y_tr == 1].mean(axis=0)       # (L, H)
        neg = H[y_tr == 0].mean(axis=0)       # (L, H)
        self.directions = pos - neg           # (L, H)
        self.midpoints = (pos + neg) / 2      # (L, H)

        # Calibrate logistic over the mean score across layers.
        scores = self._raw_score(idx_train)
        from sklearn.linear_model import LogisticRegression
        self._cal = LogisticRegression(max_iter=200, class_weight="balanced")
        self._cal.fit(scores.reshape(-1, 1), y_tr)
        return self

    def _raw_score(self, idx: np.ndarray) -> np.ndarray:
        H = self._stack(idx)
        # per-layer score, then mean
        eps = 1e-8
        d = self.directions / (np.linalg.norm(self.directions, axis=1, keepdims=True) + eps)
        s = ((H - self.midpoints) * d).sum(axis=2)  # (n, L)
        return s.mean(axis=1)                        # (n,)

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        s = self._raw_score(idx).reshape(-1, 1)
        return self._cal.predict_proba(s)

    def predict(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba(idx)[:, 1] >= 0.5


# ---------------------------------------------------------------------------
# Simple MLP probe (PyTorch)
# ---------------------------------------------------------------------------
def mlp_train_eval(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    hidden: int = 128, lr: float = 1e-3, wd: float = 1e-3, epochs: int = 150,
    seed: int = SEED,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    net = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr = torch.from_numpy(X_tr).float()
    ytr = torch.from_numpy(y_tr.astype(np.float32))
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = net(Xtr).squeeze(-1)
        loss = crit(logits, ytr)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        Xva = torch.from_numpy(X_va).float()
        logits = net(Xva).squeeze(-1)
        prob = torch.sigmoid(logits).numpy()
    pred = (prob >= 0.5).astype(int)
    return accuracy_score(y_va, pred), roc_auc_score(y_va, prob)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def eval_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple[float, float, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs, aurocs, f1s = [], [], []
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xva = sc.transform(X[va])
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr, y[tr])
        ypred = clf.predict(Xva)
        yprob = clf.predict_proba(Xva)[:, 1]
        accs.append(accuracy_score(y[va], ypred))
        f1s.append(f1_score(y[va], ypred, zero_division=0))
        aurocs.append(roc_auc_score(y[va], yprob))
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(aurocs))


def eval_mlp(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs, aurocs = [], []
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xva = sc.transform(X[va])
        a, au = mlp_train_eval(Xtr, y[tr], Xva, y[va])
        accs.append(a)
        aurocs.append(au)
    return float(np.mean(accs)), float(np.mean(aurocs))


def eval_massmean(layers: list[int], y: np.ndarray, pool: str = "last") -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs, aurocs = [], []
    idx_full = np.arange(len(y))
    for tr, va in skf.split(idx_full, y):
        m = MassMeanProbe(layers=layers, pool=pool).fit(idx_full[tr], y)
        prob = m.predict_proba(idx_full[va])[:, 1]
        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(y[va], pred))
        aurocs.append(roc_auc_score(y[va], prob))
    return float(np.mean(accs)), float(np.mean(aurocs))


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------
def main() -> None:
    results = []

    print("\n[exp] baseline_last24_lt - current default")
    a, f, au = eval_logreg(feat_single(24, "last"), y_all)
    results.append(("baseline_last24_lt", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] best_single_lt - last token layer 14")
    a, f, au = eval_logreg(feat_single(14, "last"), y_all)
    results.append(("best_single_lt", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] multi_last_lt - concat layers [10,12,14,16,18]")
    a, f, au = eval_logreg(feat_multi([10, 12, 14, 16, 18], "last"), y_all)
    results.append(("multi_last_lt_5", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] multi_all_lt - concat ALL last-token layers")
    a, f, au = eval_logreg(feat_multi(list(range(n_layers)), "last"), y_all)
    results.append(("multi_all_lt", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] mass_mean_l14 - Geometry of Truth on layer 14")
    a, au = eval_massmean([14], y_all, pool="last")
    results.append(("mass_mean_l14", a, float("nan"), au))
    print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] mass_mean_layers_avg - Mass-Mean over layers 10..18")
    a, au = eval_massmean(list(range(10, 19)), y_all, pool="last")
    results.append(("mass_mean_l10_18_avg", a, float("nan"), au))
    print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] mlp_last14 - small MLP on layer 14")
    a, au = eval_mlp(feat_single(14, "last"), y_all)
    results.append(("mlp_last14", a, float("nan"), au))
    print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] geom_only - only geometric features")
    a, f, au = eval_logreg(feat_geom(), y_all)
    results.append(("geom_only", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] mixed_lt_geom - last-token L14 + geom features")
    X14 = feat_single(14, "last")
    Xg = feat_geom()
    a, f, au = eval_logreg(np.concatenate([X14, Xg], axis=1), y_all)
    results.append(("mixed_lt14_geom", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] mixed_lt_mean14_geom - last+mean L14 + geom features")
    X14m = feat_single(14, "mean")
    a, f, au = eval_logreg(np.concatenate([X14, X14m, Xg], axis=1), y_all)
    results.append(("mixed_l14_lt+mn+geom", a, f, au))
    print(f"  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n[exp] sweep C - best regularization on layer 14 last")
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        a, f, au = eval_logreg(feat_single(14, "last"), y_all, C=C)
        print(f"  C={C:>5}  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")
        results.append((f"l14_lt_C={C}", a, f, au))

    print("\n[exp] sweep layer with strong reg C=0.1 (more reliable)")
    for layer in [10, 12, 13, 14, 15, 16, 18]:
        a, f, au = eval_logreg(feat_single(layer, "last"), y_all, C=0.1)
        print(f"  layer={layer:2d}  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")

    print("\n" + "=" * 70)
    print(f"  {'Experiment':<22} {'Acc':>7} {'F1':>7} {'AUROC':>7}")
    print("-" * 70)
    for name, a, f, au in sorted(results, key=lambda r: -r[1]):
        f_str = f"{f*100:6.2f}" if not np.isnan(f) else "    -- "
        print(f"  {name:<22} {a*100:6.2f}% {f_str}% {au*100:6.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
