"""
Tries several configurations on the response-only v2 cache:
  - resp_max at L13, L12, L17 individually with single LogReg
  - resp_max L13+L14 concat
  - resp_max + resp_mean at L13 concat
  - resp_max L13 with 5-seed bootstrap
  - LightGBM only on resp_max L13
  - LogReg + LGBM hybrid on resp_max L13
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

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

print(f"[data] N={N}")


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def cv_eval(get_probs, X, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr_idx, te_idx in skf.split(X, y):
        # Inner threshold tuning
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        i_tr, i_va = next(inner.split(X[tr_idx], y[tr_idx]))
        prob_va = get_probs(X[tr_idx][i_tr], y[tr_idx][i_tr], X[tr_idx][i_va])
        t = best_acc_threshold(prob_va, y[tr_idx][i_va])
        prob_te = get_probs(X[tr_idx], y[tr_idx], X[te_idx])
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
    return np.mean(accs), np.mean(aurocs)


def logreg_probs(Xtr, ytr, Xte, C=0.01):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def lgb_probs(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=20, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.5,
        random_state=42, verbose=-1,
    )
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def boot_logreg(Xtr, ytr, Xte, seeds=(0, 1, 7, 42, 123), C=0.01):
    probs = np.zeros(Xte.shape[0])
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        probs += logreg_probs(Xtr[idx], ytr[idx], Xte, C=C)
    return probs / len(seeds)


def hybrid(Xtr, ytr, Xte):
    return (logreg_probs(Xtr, ytr, Xte) + lgb_probs(Xtr, ytr, Xte)) / 2.0


def get_X(pool: str, layer: int) -> np.ndarray:
    return V2[f"train_{pool}"][:, layer, :].astype(np.float32)


def concat(*Xs):
    return np.concatenate(Xs, axis=1)


def main() -> None:
    EXPERIMENTS = [
        ("baseline v1: full_last L14    ", V1["train_last"][:, 14, :].astype(np.float32),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_max  L13              ", get_X("resp_max", 13),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_max  L12              ", get_X("resp_max", 12),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_max  L17              ", get_X("resp_max", 17),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_mean L13              ", get_X("resp_mean", 13),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_max L12+L13 concat    ", concat(get_X("resp_max", 12), get_X("resp_max", 13)),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_max+resp_mean L13     ", concat(get_X("resp_max", 13), get_X("resp_mean", 13)),
         lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte)),
        ("v2: resp_max L13 + boot 5 seeds", get_X("resp_max", 13),
         lambda Xtr, ytr, Xte: boot_logreg(Xtr, ytr, Xte)),
    ]
    if HAS_LGB:
        EXPERIMENTS += [
            ("v2: resp_max L13 LGBM only    ", get_X("resp_max", 13),
             lambda Xtr, ytr, Xte: lgb_probs(Xtr, ytr, Xte)),
            ("v2: resp_max L13 LR+LGB hybrid", get_X("resp_max", 13),
             lambda Xtr, ytr, Xte: hybrid(Xtr, ytr, Xte)),
        ]

    print(f"\n{'experiment':<35} {'acc':>7} {'auroc':>7}")
    print("-" * 60)
    rows = []
    for name, X, fn in EXPERIMENTS:
        a, au = cv_eval(fn, X, y)
        rows.append((name, a, au))
        print(f"  {name:<33} {a*100:6.2f}% {au*100:6.2f}%")

    print("\n=== Stability of best on 7 seeds ===")
    best_X = get_X("resp_max", 13)
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, au = cv_eval(lambda Xtr, ytr, Xte: logreg_probs(Xtr, ytr, Xte),
                        best_X, y, seed=seed)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
