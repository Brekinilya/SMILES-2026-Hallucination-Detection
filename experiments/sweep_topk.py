"""
For each layer in {11..15} and K in {1, 3, 5, 10, 20}, run LogReg with
5-seed bootstrap on the v4 cache and print accuracy and AUROC. K=1 is
plain max. Used to check if top-K helps on the production setup.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
V1 = np.load(REPO_ROOT / "cache" / "features.npz")
V4 = np.load(REPO_ROOT / "cache" / "features_v4.npz")
y = V1["train_labels"].astype(int)
N = len(y)
LAYERS = list(V4["layers"])
KS = list(V4["ks"])
print(f"[data] N={N}  layers={LAYERS}  ks={KS}")


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def fit_logreg(Xtr, ytr, C=0.01):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def boot_probs(Xtr, ytr, Xte, seeds=(0, 1, 7, 42, 123)):
    p = np.zeros(Xte.shape[0])
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        sc, clf = fit_logreg(Xtr[idx], ytr[idx])
        p += clf.predict_proba(sc.transform(Xte))[:, 1]
    return p / len(seeds)


def cv_eval(X, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr_idx, te_idx in skf.split(X, y):
        oof = np.zeros(len(tr_idx))
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
        for in_tr, in_va in inner.split(np.arange(len(tr_idx)), y[tr_idx]):
            oof[in_va] = boot_probs(X[tr_idx][in_tr], y[tr_idx][in_tr], X[tr_idx][in_va])
        t = best_acc_threshold(oof, y[tr_idx])
        prob_te = boot_probs(X[tr_idx], y[tr_idx], X[te_idx])
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
    return np.mean(accs), np.mean(aurocs)


def main() -> None:
    L_IDX = {l: i for i, l in enumerate(LAYERS)}

    print(f"\n{'layer':>5}  ", end="")
    for k in KS:
        print(f" {'K='+str(k):>14}", end="")
    print()
    print("-" * (5 + 2 + 16 * len(KS)))

    rows = []
    for layer in LAYERS:
        print(f"  L{int(layer):<3}  ", end="")
        for k in KS:
            X = V4[f"train_topk{k}"][:, L_IDX[layer], :].astype(np.float32)
            a, au = cv_eval(X, y)
            rows.append((int(layer), int(k), a, au))
            print(f" {a*100:5.2f}/{au*100:5.2f} ", end="")
        print()

    print("\nTop 8 by accuracy:")
    rows.sort(key=lambda r: -r[2])
    for l, k, a, au in rows[:8]:
        print(f"  L{l:>2} K={k:<3}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
