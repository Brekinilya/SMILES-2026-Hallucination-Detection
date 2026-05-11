"""
On the v1 (full-sequence) features:
  1. Multi-seed bootstrap ensemble of LogReg, probabilities averaged.
  2. LightGBM head on the same features.
  3. Independent LogReg per layer (L13..L16), probabilities averaged.
  4. LogReg + LGBM hybrid (probabilities averaged).
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

y = V1["train_labels"].astype(int)
N = len(y)
print(f"[data] N={N}")

# Try to import lightgbm; if unavailable, skip those experiments.
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not installed; LGB experiments will be skipped")


def make_X_layer(layer: int, pool: str = "last") -> np.ndarray:
    return V1[f"train_{pool}"][:, layer, :].astype(np.float32)


def best_acc_threshold(probs, y_true):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Single-model probe factories
# ---------------------------------------------------------------------------
def fit_logreg(Xtr, ytr, C=0.01, seed=42):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
    clf.fit(sc.transform(Xtr), ytr)
    return ("logreg", sc, clf)


def fit_lgb(Xtr, ytr, seed=42):
    sc = StandardScaler().fit(Xtr)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(sc.transform(Xtr), ytr)
    return ("lgb", sc, clf)


def predict_proba(model_tuple, X):
    _, sc, clf = model_tuple
    return clf.predict_proba(sc.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------
def run_cv_ensemble(get_probs, X, y, n_folds=5, seed=42):
    """get_probs(Xtr, ytr, Xva) -> probs_va.   threshold tuned on inner split."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr_idx, va_idx in skf.split(X, y):
        # Inner split for threshold
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        i_tr, i_va = next(inner.split(X[tr_idx], y[tr_idx]))
        prob_va_inner = get_probs(X[tr_idx][i_tr], y[tr_idx][i_tr], X[tr_idx][i_va])
        t = best_acc_threshold(prob_va_inner, y[tr_idx][i_va])
        # Refit on full train for outer test prediction
        prob = get_probs(X[tr_idx], y[tr_idx], X[va_idx])
        pred = (prob >= t).astype(int)
        accs.append(accuracy_score(y[va_idx], pred))
        aurocs.append(roc_auc_score(y[va_idx], prob))
    return np.mean(accs), np.mean(aurocs)


def multiseed_logreg(Xtr, ytr, Xte, seeds=(0, 1, 7, 42, 123, 2024, 31415), C=0.01):
    """Average probabilities across LogReg fits with different seeds.
    Note: for sklearn LogReg with lbfgs, the objective is convex so seed
    doesn't change the *fit*, but bootstrap-resample the training set so
    we get genuine model diversity."""
    rng = np.random.RandomState(42)
    probs = np.zeros(Xte.shape[0])
    for s in seeds:
        # Bootstrap resample
        rng_s = np.random.RandomState(s)
        idx = rng_s.choice(len(Xtr), size=len(Xtr), replace=True)
        m = fit_logreg(Xtr[idx], ytr[idx], C=C)
        probs += predict_proba(m, Xte)
    return probs / len(seeds)


def multilayer_logreg(Xtr_dict, ytr, Xte_dict, C=0.01):
    """Average probs across LogReg fits on different layers."""
    layers = list(Xtr_dict.keys())
    probs = np.zeros(next(iter(Xte_dict.values())).shape[0])
    for l in layers:
        m = fit_logreg(Xtr_dict[l], ytr, C=C)
        probs += predict_proba(m, Xte_dict[l])
    return probs / len(layers)


def hybrid_logreg_lgb(Xtr, ytr, Xte, C=0.01):
    m1 = fit_logreg(Xtr, ytr, C=C)
    m2 = fit_lgb(Xtr, ytr)
    return (predict_proba(m1, Xte) + predict_proba(m2, Xte)) / 2.0


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
def main() -> None:
    # Reference: single LogReg L14 C=0.01 (currently in production)
    print("\n=== Reference: LogReg L14 last C=0.01 ===")
    a, au = run_cv_ensemble(lambda Xtr, ytr, Xte: predict_proba(fit_logreg(Xtr, ytr), Xte),
                            make_X_layer(14, "last"), y)
    print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")

    print("\n=== Multi-seed bootstrap ensemble (7 seeds) on L14 last ===")
    a, au = run_cv_ensemble(lambda Xtr, ytr, Xte: multiseed_logreg(Xtr, ytr, Xte),
                            make_X_layer(14, "last"), y)
    print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")

    print("\n=== Multi-layer ensemble: independent LogReg on L13,14,15,16 ===")
    Xs_train = {l: make_X_layer(l, "last") for l in [13, 14, 15, 16]}
    def ml_proba(Xtr_dummy, ytr, Xte_dummy):
        # Need per-layer slices, ignore the X passed by the harness
        # We'll select indices from the train fold and test fold ourselves
        raise NotImplementedError
    # Run multi-layer manually for clarity
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for tr_idx, te_idx in skf.split(np.arange(N), y):
        # Inner threshold tuning
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=43)
        i_tr, i_va = next(inner.split(np.arange(len(tr_idx)), y[tr_idx]))
        # Compute multilayer probs on inner-val for threshold
        Xtr_dict_inner = {l: Xs_train[l][tr_idx][i_tr] for l in Xs_train}
        Xva_dict_inner = {l: Xs_train[l][tr_idx][i_va] for l in Xs_train}
        prob_va_inner = multilayer_logreg(Xtr_dict_inner, y[tr_idx][i_tr], Xva_dict_inner)
        t = best_acc_threshold(prob_va_inner, y[tr_idx][i_va])
        # Final per-fold: refit on full outer train
        Xtr_dict_full = {l: Xs_train[l][tr_idx] for l in Xs_train}
        Xte_dict = {l: Xs_train[l][te_idx] for l in Xs_train}
        prob = multilayer_logreg(Xtr_dict_full, y[tr_idx], Xte_dict)
        pred = (prob >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        aurocs.append(roc_auc_score(y[te_idx], prob))
    print(f"  acc={np.mean(accs)*100:.2f}  auroc={np.mean(aurocs)*100:.2f}")

    if HAS_LGB:
        print("\n=== LightGBM only on L14 last ===")
        a, au = run_cv_ensemble(lambda Xtr, ytr, Xte: predict_proba(fit_lgb(Xtr, ytr), Xte),
                                make_X_layer(14, "last"), y)
        print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")

        print("\n=== Hybrid LogReg + LGBM on L14 last ===")
        a, au = run_cv_ensemble(lambda Xtr, ytr, Xte: hybrid_logreg_lgb(Xtr, ytr, Xte),
                                make_X_layer(14, "last"), y)
        print(f"  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
