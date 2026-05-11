"""
Two extensions to the single-L13 + fixed-C=0.01 production:
  1. Concat layer 12 + layer 13 response-only max-pool (1792-dim).
  2. Adaptive C selection in {0.003, 0.01, 0.03, 0.1} via OOF on training data.

Test stability across 7 seeds to see if they reliably beat the production.
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
V2 = np.load(REPO_ROOT / "cache" / "features_v2.npz")
y = V1["train_labels"].astype(int)
N = len(y)

CANDIDATE_C = (0.003, 0.01, 0.03, 0.1)


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def fit_logreg(Xtr, ytr, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def boot_probs_fixed_C(Xtr, ytr, Xte, C, seeds=(0, 1, 7, 42, 123)):
    p = np.zeros(Xte.shape[0])
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        sc, clf = fit_logreg(Xtr[idx], ytr[idx], C=C)
        p += clf.predict_proba(sc.transform(Xte))[:, 1]
    return p / len(seeds)


def select_c(Xtr, ytr, n_folds=5, seed=42):
    """Pick C from CANDIDATE_C using stratified OOF accuracy."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    best_C, best_oof_acc = CANDIDATE_C[0], -1.0
    for C in CANDIDATE_C:
        oof = np.zeros(len(ytr))
        for tr, va in skf.split(np.arange(len(ytr)), ytr):
            sc, clf = fit_logreg(Xtr[tr], ytr[tr], C=C)
            oof[va] = clf.predict_proba(sc.transform(Xtr[va]))[:, 1]
        t = best_acc_threshold(oof, ytr)
        oof_acc = accuracy_score(ytr, (oof >= t).astype(int))
        if oof_acc > best_oof_acc:
            best_oof_acc, best_C = oof_acc, C
    return best_C


def cv_eval(get_probs, X, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr_idx, te_idx in skf.split(X, y):
        oof = np.zeros(len(tr_idx))
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
        for in_tr, in_va in inner.split(np.arange(len(tr_idx)), y[tr_idx]):
            oof[in_va] = get_probs(X[tr_idx][in_tr], y[tr_idx][in_tr], X[tr_idx][in_va])
        t = best_acc_threshold(oof, y[tr_idx])
        prob_te = get_probs(X[tr_idx], y[tr_idx], X[te_idx])
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
    return np.mean(accs), np.mean(aurocs)


L13 = V2["train_resp_max"][:, 13, :].astype(np.float32)
L12_13 = np.concatenate([
    V2["train_resp_max"][:, 12, :].astype(np.float32),
    V2["train_resp_max"][:, 13, :].astype(np.float32),
], axis=1)
print(f"L13 shape: {L13.shape}  L12+L13 shape: {L12_13.shape}")


def main() -> None:
    # Run a configuration across 7 seeds and report mean ± std.
    SETUPS = [
        ("L13 fixed C=0.01 (production)", L13, "fixed", 0.01),
        ("L13 C-search",                   L13, "search", None),
        ("L12+L13 fixed C=0.01",           L12_13, "fixed", 0.01),
        ("L12+L13 C-search",               L12_13, "search", None),
    ]

    print(f"\n{'config':<35} {'mean acc':>10} {'std':>6} {'mean auroc':>11}")
    print("-" * 75)
    SEEDS = [0, 1, 7, 42, 123, 2024, 31415]
    for name, X, mode, C_fixed in SETUPS:
        accs, aurocs = [], []
        for seed in SEEDS:
            if mode == "fixed":
                fn = lambda Xtr, ytr, Xte: boot_probs_fixed_C(Xtr, ytr, Xte, C=C_fixed)
            else:
                # search C inside the harness using *training-fold* OOF
                def fn(Xtr, ytr, Xte, _seed=seed):
                    C = select_c(Xtr, ytr, seed=_seed + 99)
                    return boot_probs_fixed_C(Xtr, ytr, Xte, C=C)
            a, au = cv_eval(fn, X, y, seed=seed)
            accs.append(a); aurocs.append(au)
        print(f"  {name:<33} {np.mean(accs)*100:7.2f}% {np.std(accs)*100:5.2f}  {np.mean(aurocs)*100:8.2f}%")

    # Per-seed for the new candidate L12+L13 C-search
    print("\nPer-seed for L12+L13 C-search:")
    for seed in SEEDS:
        def fn(Xtr, ytr, Xte, _seed=seed):
            C = select_c(Xtr, ytr, seed=_seed + 99)
            return boot_probs_fixed_C(Xtr, ytr, Xte, C=C)
        a, au = cv_eval(fn, L12_13, y, seed=seed)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
