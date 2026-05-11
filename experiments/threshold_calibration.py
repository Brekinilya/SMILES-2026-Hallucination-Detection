"""
Test alternative threshold strategies. Production uses accuracy-tuned on
OOF, which on the held-out test.csv ends up predicting 88% positives vs
the training prior of 70%. Try a few variants to see if matching the
prior helps or hurts CV accuracy.

Strategies compared:
  - accuracy-tuned on OOF (production)
  - quantile threshold so the predicted positive rate matches 70%
  - midpoint of the two
  - accuracy-tuned plus a fixed offset
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
TRAIN_POS_RATE = float(y.mean())  # 0.701
print(f"[data] N={N}  train_positive_rate={TRAIN_POS_RATE*100:.2f}%")


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def quantile_threshold(probs, target_pos_rate=TRAIN_POS_RATE):
    """Pick t such that (probs >= t).mean() == target_pos_rate."""
    return float(np.quantile(probs, 1.0 - target_pos_rate))


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


def cv_eval_threshold(X, y, threshold_strategy: str, n_folds=5, seed=42):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    accs, aurocs, ts, pos_rates = [], [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        oof = np.zeros(len(tr_idx))
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
        for in_tr, in_va in inner.split(np.arange(len(tr_idx)), y[tr_idx]):
            oof[in_va] = boot_probs(X[tr_idx][in_tr], y[tr_idx][in_tr], X[tr_idx][in_va])
        if threshold_strategy == "acc":
            t = best_acc_threshold(oof, y[tr_idx])
        elif threshold_strategy == "prior":
            t = quantile_threshold(oof, target_pos_rate=TRAIN_POS_RATE)
        elif threshold_strategy == "mid":
            t = 0.5 * best_acc_threshold(oof, y[tr_idx]) + 0.5 * quantile_threshold(oof)
        elif threshold_strategy.startswith("acc+"):
            # acc-tuned plus a fixed offset, e.g. "acc+0.05"
            offset = float(threshold_strategy.split("+")[1])
            t = best_acc_threshold(oof, y[tr_idx]) + offset
        else:
            raise ValueError(threshold_strategy)
        prob_te = boot_probs(X[tr_idx], y[tr_idx], X[te_idx])
        pred = (prob_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob_te))
        ts.append(t)
        pos_rates.append(float(pred.mean()))
    return (np.mean(accs), np.mean(aurocs),
            np.mean(ts), np.mean(pos_rates))


def main() -> None:
    X = V2["train_resp_max"][:, 13, :].astype(np.float32)

    EXPS = ["acc", "prior", "mid", "acc+0.03", "acc+0.05", "acc+0.10"]
    print(f"\n{'strategy':<14} {'acc':>7} {'auroc':>7} {'thresh':>7} {'pos_rate':>9}")
    print("-" * 50)
    for s in EXPS:
        a, au, t, pr = cv_eval_threshold(X, y, s)
        print(f"  {s:<12} {a*100:6.2f}% {au*100:6.2f}% {t:6.3f}  {pr*100:6.2f}%")

    # Stability of best across seeds
    print("\n=== Stability of strategy='prior' across 7 seeds ===")
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, au, t, pr = cv_eval_threshold(X, y, "prior", seed=seed)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}  pos_rate={pr*100:.1f}%")

    print("\n=== Stability of strategy='acc' (current production) across 7 seeds ===")
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, au, t, pr = cv_eval_threshold(X, y, "acc", seed=seed)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}  pos_rate={pr*100:.1f}%")


if __name__ == "__main__":
    main()
