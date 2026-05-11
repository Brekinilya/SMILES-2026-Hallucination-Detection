"""
Two ideas tested on the v2 cache:
  A. Concat of three pools at L13: max + mean + first response token.
  B. Isotonic calibration of OOF probabilities before accuracy threshold.
Both compared to production (single-L13 resp_max with 5-seed bootstrap).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
V1 = np.load(REPO_ROOT / "cache" / "features.npz")
V2 = np.load(REPO_ROOT / "cache" / "features_v2.npz")
y = V1["train_labels"].astype(int)
N = len(y)

print(f"[data] N={N}")


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


def boot_probs(Xtr, ytr, Xte, seeds=(0, 1, 7, 42, 123), C=0.01):
    p = np.zeros(Xte.shape[0])
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        sc, clf = fit_logreg(Xtr[idx], ytr[idx], C=C)
        p += clf.predict_proba(sc.transform(Xte))[:, 1]
    return p / len(seeds)


def cv_eval(get_probs, X, y, n_folds=5, seed=42, calibrator=None):
    """Optional `calibrator(probs_oof, y_train)` returns a fitted callable
    used to remap probabilities at test time."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr_idx, te_idx in skf.split(X, y):
        # OOF probabilities on the outer-train set for threshold tuning + calibration
        oof = np.zeros(len(tr_idx))
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        for in_tr, in_va in inner.split(np.arange(len(tr_idx)), y[tr_idx]):
            oof[in_va] = get_probs(X[tr_idx][in_tr], y[tr_idx][in_tr], X[tr_idx][in_va])
        if calibrator is not None:
            cal = calibrator(oof, y[tr_idx])
            oof_calibrated = cal(oof)
            t = best_acc_threshold(oof_calibrated, y[tr_idx])
        else:
            cal = None
            t = best_acc_threshold(oof, y[tr_idx])
        # Outer-test probability
        prob_te = get_probs(X[tr_idx], y[tr_idx], X[te_idx])
        if cal is not None:
            prob_te = cal(prob_te)
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
    return np.mean(accs), np.mean(aurocs)


def isotonic_calibrator(oof, y_tr):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof, y_tr)
    return lambda p: iso.predict(p)


def make_X_v2(pool: str, layer: int) -> np.ndarray:
    return V2[f"train_{pool}"][:, layer, :].astype(np.float32)


def main() -> None:
    L13_max = make_X_v2("resp_max", 13)
    L13_mean = make_X_v2("resp_mean", 13)
    L13_first = make_X_v2("resp_first", 13)

    EXPS = [
        ("PROD: resp_max L13 + boot",      L13_max,                     None),
        ("PROD + isotonic calibration",    L13_max,                     isotonic_calibrator),
        ("triple concat: max+mean+first L13", np.concatenate([L13_max, L13_mean, L13_first], axis=1), None),
        ("triple + isotonic",              np.concatenate([L13_max, L13_mean, L13_first], axis=1), isotonic_calibrator),
        ("max+mean concat L13",            np.concatenate([L13_max, L13_mean], axis=1), None),
        ("max+first concat L13",           np.concatenate([L13_max, L13_first], axis=1), None),
    ]

    print(f"\n{'experiment':<42} {'acc':>7} {'auroc':>7}")
    print("-" * 65)
    for name, X, cal in EXPS:
        a, au = cv_eval(lambda Xtr, ytr, Xte: boot_probs(Xtr, ytr, Xte), X, y, calibrator=cal)
        print(f"  {name:<40} {a*100:6.2f}% {au*100:6.2f}%")

    # Stability of best across seeds
    print("\n=== Stability across 7 seeds: PROD + isotonic ===")
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, au = cv_eval(lambda Xtr, ytr, Xte: boot_probs(Xtr, ytr, Xte),
                        L13_max, y, seed=seed, calibrator=isotonic_calibrator)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
