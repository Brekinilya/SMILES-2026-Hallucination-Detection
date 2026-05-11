"""
For every (layer, pool) pair on the full-sequence cache, fit LogReg with
5-fold stratified CV and print mean accuracy and AUROC.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO_ROOT / "cache" / "features.npz"
OUT_DIR = REPO_ROOT / "experiments" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POOLS = ["last", "mean", "max"]
N_FOLDS = 5
RANDOM_STATE = 42


def evaluate_layer_pool(
    X: np.ndarray, y: np.ndarray, layer: int
) -> tuple[float, float]:
    """5-fold stratified CV; mean (accuracy, AUROC) on validation folds."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, aurocs = [], []
    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(Xtr, y[tr])
        ypred = clf.predict(Xva)
        yprob = clf.predict_proba(Xva)[:, 1]
        accs.append(accuracy_score(y[va], ypred))
        aurocs.append(roc_auc_score(y[va], yprob))
    return float(np.mean(accs)), float(np.mean(aurocs))


def main() -> None:
    print(f"[load] {CACHE_FILE}")
    cache = np.load(CACHE_FILE)
    y = cache["train_labels"].astype(int)
    n = len(y)
    print(f"[data] N={n}  pos={int(y.sum())}  neg={int((y==0).sum())}")
    print(f"       baseline (always 1) acc = {y.mean()*100:.2f}%")

    # All pool arrays have shape (N, n_layers, hidden_dim)
    pool_arrs = {p: cache[f"train_{p}"].astype(np.float32) for p in POOLS}
    n_layers = pool_arrs["last"].shape[1]
    print(f"[feat] layers={n_layers}  hidden_dim={pool_arrs['last'].shape[2]}")

    acc_mat = np.zeros((len(POOLS), n_layers))
    auroc_mat = np.zeros((len(POOLS), n_layers))

    for pi, pool in enumerate(POOLS):
        for layer in range(n_layers):
            X = pool_arrs[pool][:, layer, :]
            acc, auroc = evaluate_layer_pool(X, y, layer)
            acc_mat[pi, layer] = acc
            auroc_mat[pi, layer] = auroc
            print(f"  pool={pool:4s} layer={layer:2d}  acc={acc*100:5.2f}%  auroc={auroc*100:5.2f}%")

    # Save raw matrices
    np.savez(
        OUT_DIR / "layer_pool_sweep.npz",
        acc=acc_mat, auroc=auroc_mat, pools=np.array(POOLS),
    )

    print("\n[summary] best by accuracy:")
    flat_acc = [(POOLS[pi], li, acc_mat[pi, li], auroc_mat[pi, li])
                for pi in range(len(POOLS)) for li in range(n_layers)]
    flat_acc.sort(key=lambda x: -x[2])
    for p, l, a, au in flat_acc[:10]:
        print(f"  pool={p:4s} layer={l:2d}  acc={a*100:5.2f}%  auroc={au*100:5.2f}%")

    print("\n[summary] best by AUROC:")
    flat_acc.sort(key=lambda x: -x[3])
    for p, l, a, au in flat_acc[:10]:
        print(f"  pool={p:4s} layer={l:2d}  acc={a*100:5.2f}%  auroc={au*100:5.2f}%")


if __name__ == "__main__":
    main()
