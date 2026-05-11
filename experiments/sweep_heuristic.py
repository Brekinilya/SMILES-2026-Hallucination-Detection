"""
Compare exact response pooling (v2) against the last-K-percent heuristic
(v3). If the heuristic is close to exact, I can skip the tokenizer load
inside aggregation.py. Result: heuristic loses 2 to 7 AUROC, so I keep
the exact boundary.
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
V3 = np.load(REPO_ROOT / "cache" / "features_v3.npz")

y = V1["train_labels"].astype(int)
N = len(y)
LAYERS = list(V3["layers"])
FRACS = [0.20, 0.30, 0.40, 0.50, 0.70]
print(f"[data] N={N}  layers={LAYERS}  fracs={FRACS}")


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def cv_eval_logreg(X, y, C=0.01, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr_idx, te_idx in skf.split(X, y):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        i_tr, i_va = next(inner.split(X[tr_idx], y[tr_idx]))
        sc = StandardScaler().fit(X[tr_idx][i_tr])
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
        clf.fit(sc.transform(X[tr_idx][i_tr]), y[tr_idx][i_tr])
        prob_va = clf.predict_proba(sc.transform(X[tr_idx][i_va]))[:, 1]
        t = best_acc_threshold(prob_va, y[tr_idx][i_va])
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
        clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
        prob = clf.predict_proba(sc.transform(X[te_idx]))[:, 1]
        pred = (prob >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob))
    return np.mean(accs), np.mean(aurocs)


def main() -> None:
    # Map layer index to position in V3
    L_IDX = {l: i for i, l in enumerate(LAYERS)}

    print("\n=== Heuristic last-K%-tokens (v3) sweep ===")
    print(f"{'layer':>5}  ", end="")
    for f in FRACS:
        print(f"{'f'+str(int(f*100)):>14}", end="")
    print(f"{'exact resp_max':>18}")
    for layer in LAYERS:
        layer = int(layer)
        print(f"  L{layer:<3}  ", end="")
        for f in FRACS:
            key = f"f{int(f*100):02d}"
            X = V3[f"train_{key}_max"][:, L_IDX[layer], :].astype(np.float32)
            a, au = cv_eval_logreg(X, y)
            print(f" {a*100:5.2f}/{au*100:5.2f} ", end="")
        # Exact reference from v2
        X_exact = V2["train_resp_max"][:, layer, :].astype(np.float32)
        a, au = cv_eval_logreg(X_exact, y)
        print(f"  {a*100:5.2f}/{au*100:5.2f}")

    print("\n=== Best heuristic configs (top 10 by accuracy) ===")
    rows = []
    for layer in LAYERS:
        layer = int(layer)
        for f in FRACS:
            key = f"f{int(f*100):02d}"
            for pool in ["max", "mean"]:
                X = V3[f"train_{key}_{pool}"][:, L_IDX[layer], :].astype(np.float32)
                a, au = cv_eval_logreg(X, y)
                rows.append((layer, f, pool, a, au))
    rows.sort(key=lambda r: -r[3])
    for l, f, p, a, au in rows[:10]:
        print(f"  layer={l:>2} f={f:.2f} pool={p:<4}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
