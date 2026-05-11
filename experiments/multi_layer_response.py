"""
Independent LogReg per layer on response-only features, with averaged
probabilities (no feature concat). Compares 2, 3 and 5 layer sets
against the single-L13 + bootstrap baseline.
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


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def fit_probs(X_tr, y_tr, X_te, C=0.01):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def boot(X_tr, y_tr, X_te, seeds=(0, 1, 7, 42, 123), C=0.01):
    p = np.zeros(X_te.shape[0])
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = rng.choice(len(X_tr), size=len(X_tr), replace=True)
        p += fit_probs(X_tr[idx], y_tr[idx], X_te, C=C)
    return p / len(seeds)


def multilayer_avg(layers, pool="resp_max"):
    """Returns a function (Xtr_dummy, ytr, Xte_dummy) -> probs that uses
    indices to slice per-layer arrays from V2 globally. Indices are the
    sample indices used by the cv harness."""
    Xs = {l: V2[f"train_{pool}"][:, l, :].astype(np.float32) for l in layers}
    def fn(idx_tr, ytr, idx_te):
        p = np.zeros(len(idx_te))
        for l in layers:
            p += fit_probs(Xs[l][idx_tr], ytr, Xs[l][idx_te])
        return p / len(layers)
    return fn


def multilayer_boot(layers, pool="resp_max"):
    Xs = {l: V2[f"train_{pool}"][:, l, :].astype(np.float32) for l in layers}
    def fn(idx_tr, ytr, idx_te):
        p = np.zeros(len(idx_te))
        for l in layers:
            p += boot(Xs[l][idx_tr], ytr, Xs[l][idx_te])
        return p / len(layers)
    return fn


def cv_eval_idx_based(get_probs, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    idx_full = np.arange(len(y))
    for tr_idx, te_idx in skf.split(idx_full, y):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        i_tr_local, i_va_local = next(inner.split(np.arange(len(tr_idx)), y[tr_idx]))
        i_tr = tr_idx[i_tr_local]
        i_va = tr_idx[i_va_local]
        prob_va = get_probs(i_tr, y[i_tr], i_va)
        t = best_acc_threshold(prob_va, y[i_va])
        prob_te = get_probs(tr_idx, y[tr_idx], te_idx)
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
    return np.mean(accs), np.mean(aurocs)


def main() -> None:
    print(f"[data] N={N}\n")
    print("Multi-layer ensembles on response-only max-pool:")

    EXPS = [
        ("L13 single", multilayer_avg([13])),
        ("L13 boot",   multilayer_boot([13])),
        ("L12+L13 avg",          multilayer_avg([12, 13])),
        ("L12+L13 boot avg",     multilayer_boot([12, 13])),
        ("L12+L13+L17 avg",      multilayer_avg([12, 13, 17])),
        ("L12+L13+L17 boot avg", multilayer_boot([12, 13, 17])),
        ("L12+L13+L14+L15+L17 avg",  multilayer_avg([12, 13, 14, 15, 17])),
        ("L12+L13+L14+L15+L17 boot", multilayer_boot([12, 13, 14, 15, 17])),
    ]

    print(f"{'experiment':<35} {'acc':>7} {'auroc':>7}")
    print("-" * 60)
    for name, fn in EXPS:
        a, au = cv_eval_idx_based(fn, y)
        print(f"  {name:<33} {a*100:6.2f}% {au*100:6.2f}%")


if __name__ == "__main__":
    main()
