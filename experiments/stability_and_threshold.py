"""
Sanity checks on the iter-1 setup.

1. Layer sweep over 7 seeds to confirm L14 wins not just on seed=42.
2. class_weight balanced vs None under several threshold strategies.
3. Threshold at 0.5 vs validation-tuned for accuracy vs for F1.

Uses nested CV: outer 5-fold for evaluation, inner split for threshold
tuning.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = np.load(REPO_ROOT / "cache" / "features.npz")
y = CACHE["train_labels"].astype(int)
N = len(y)
print(f"[data] N={N}  baseline_acc={y.mean()*100:.2f}%")


def best_threshold(probs: np.ndarray, y_true: np.ndarray, metric: str = "acc") -> float:
    """Pick the threshold on (probs, y_true) that maximizes the chosen metric."""
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_v = 0.5, -1.0
    for t in cand:
        pred = (probs >= t).astype(int)
        if metric == "acc":
            v = accuracy_score(y_true, pred)
        elif metric == "f1":
            v = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(metric)
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t


def fit_logreg(X_tr, y_tr, C=1.0, class_weight=None):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight=class_weight, solver="lbfgs")
    clf.fit(Xs, y_tr)
    return sc, clf


def eval_probe(
    X: np.ndarray, y: np.ndarray, layer: int, C: float, class_weight,
    threshold_metric: str | None, n_folds: int = 5, seed: int = 42,
):
    """Outer 5-fold; on each fold do an inner train/val split for threshold tuning."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, f1s, aurocs = [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        if threshold_metric is None:
            sc, clf = fit_logreg(X[tr_idx], y[tr_idx], C=C, class_weight=class_weight)
            t = 0.5
        else:
            # nested split for threshold
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
            i_tr, i_va = next(inner.split(X[tr_idx], y[tr_idx]))
            sc, clf = fit_logreg(
                X[tr_idx][i_tr], y[tr_idx][i_tr], C=C, class_weight=class_weight
            )
            probs_va = clf.predict_proba(sc.transform(X[tr_idx][i_va]))[:, 1]
            t = best_threshold(probs_va, y[tr_idx][i_va], metric=threshold_metric)
            # refit on full outer train for final probe
            sc, clf = fit_logreg(X[tr_idx], y[tr_idx], C=C, class_weight=class_weight)

        probs = clf.predict_proba(sc.transform(X[te_idx]))[:, 1]
        pred = (probs >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        f1s.append(f1_score(y[te_idx], pred, zero_division=0))
        aurocs.append(roc_auc_score(y[te_idx], probs))
    return np.mean(accs), np.mean(f1s), np.mean(aurocs)


def main() -> None:
    print("\n=== Stability across seeds (layer 14, C=1.0, balanced, t=0.5) ===")
    X14 = CACHE["train_last"][:, 14, :].astype(np.float32)
    seed_results = []
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, f, au = eval_probe(X14, y, 14, C=1.0, class_weight="balanced",
                              threshold_metric=None, seed=seed)
        seed_results.append((seed, a, f, au))
        print(f"  seed={seed:6d}  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")
    accs = [r[1] for r in seed_results]
    aurocs = [r[3] for r in seed_results]
    print(f"  mean: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}  "
          f"auroc={np.mean(aurocs)*100:.2f}±{np.std(aurocs)*100:.2f}")

    print("\n=== Layer x class_weight x threshold (seed=42) ===")
    print(f"  {'layer':>5} {'cw':>10} {'thresh':>8} {'acc':>7} {'f1':>7} {'auroc':>7}")
    for layer in [12, 13, 14, 15, 16]:
        Xl = CACHE["train_last"][:, layer, :].astype(np.float32)
        for cw in ["balanced", None]:
            for tm in [None, "acc", "f1"]:
                a, f, au = eval_probe(Xl, y, layer, C=1.0, class_weight=cw,
                                      threshold_metric=tm)
                tm_str = tm if tm else "0.5"
                cw_str = "bal" if cw == "balanced" else "none"
                print(f"  {layer:>5} {cw_str:>10} {tm_str:>8} "
                      f"{a*100:6.2f}% {f*100:6.2f}% {au*100:6.2f}%")

    print("\n=== Best C for layer 14 with cw=None and acc-tuned threshold ===")
    Xl = CACHE["train_last"][:, 14, :].astype(np.float32)
    for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        a, f, au = eval_probe(Xl, y, 14, C=C, class_weight=None, threshold_metric="acc")
        print(f"  C={C:>6}  acc={a*100:.2f}  f1={f*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
