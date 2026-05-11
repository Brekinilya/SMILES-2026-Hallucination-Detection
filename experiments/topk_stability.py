"""
Stability of top-K=10 max-pool at L13 across 7 different CV seeds.
Also: does combining K=10 with multi-layer averaging help further?
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
L_IDX = {int(l): i for i, l in enumerate(LAYERS)}


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


def cv_eval(get_probs, y, n_folds=5, seed=42):
    """get_probs(idx_tr, ytr, idx_te) -> probs."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    idx_full = np.arange(len(y))
    for tr_idx, te_idx in skf.split(idx_full, y):
        oof = np.zeros(len(tr_idx))
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
        for in_tr, in_va in inner.split(np.arange(len(tr_idx)), y[tr_idx]):
            oof[in_va] = get_probs(tr_idx[in_tr], y[tr_idx[in_tr]], tr_idx[in_va])
        t = best_acc_threshold(oof, y[tr_idx])
        prob_te = get_probs(tr_idx, y[tr_idx], te_idx)
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
    return np.mean(accs), np.mean(aurocs)


def single_layer_topk(layer, k):
    X = V4[f"train_topk{k}"][:, L_IDX[layer], :].astype(np.float32)
    def fn(idx_tr, ytr, idx_te):
        return boot_probs(X[idx_tr], ytr, X[idx_te])
    return fn


def multilayer_topk_avg(layers, k):
    Xs = {l: V4[f"train_topk{k}"][:, L_IDX[l], :].astype(np.float32) for l in layers}
    def fn(idx_tr, ytr, idx_te):
        p = np.zeros(len(idx_te))
        for l in layers:
            p += boot_probs(Xs[l][idx_tr], ytr, Xs[l][idx_te])
        return p / len(layers)
    return fn


def main() -> None:
    print(f"[data] N={N}\n")

    print("=== Stability of L13 top-10 + bootstrap across 7 seeds ===")
    accs, aurocs = [], []
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, au = cv_eval(single_layer_topk(13, 10), y, seed=seed)
        accs.append(a); aurocs.append(au)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}")
    print(f"  mean: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}  "
          f"auroc={np.mean(aurocs)*100:.2f}±{np.std(aurocs)*100:.2f}")

    # Compare to vanilla max
    print("\n=== Stability of L13 top-1 (max) for reference ===")
    accs, aurocs = [], []
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, au = cv_eval(single_layer_topk(13, 1), y, seed=seed)
        accs.append(a); aurocs.append(au)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}")
    print(f"  mean: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}  "
          f"auroc={np.mean(aurocs)*100:.2f}±{np.std(aurocs)*100:.2f}")

    # Multi-layer averaging on top-K
    print("\n=== Multi-layer top-10 ensembles ===")
    EXPS = [
        ("L13 only top-10",            single_layer_topk(13, 10)),
        ("L12+L13 top-10 avg",         multilayer_topk_avg([12, 13], 10)),
        ("L12+L13+L14 top-10 avg",     multilayer_topk_avg([12, 13, 14], 10)),
        ("L11+L12+L13+L14+L15 top-10", multilayer_topk_avg([11, 12, 13, 14, 15], 10)),
    ]
    for name, fn in EXPS:
        a, au = cv_eval(fn, y)
        print(f"  {name:<32}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
