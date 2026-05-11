"""
Same sweep as sweep_layer_pool.py but on the response-only cache (v2).

Compares resp_first, resp_mean, resp_max per layer, with full_last from
v1 as a reference. Response-only max wins at layer 13.
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
n_layers = V2["train_resp_mean"].shape[1]
print(f"[data] N={N}  layers={n_layers}")
print(f"[data] resp_lens: min={V2['train_resp_lens'].min()} median={int(np.median(V2['train_resp_lens']))} max={V2['train_resp_lens'].max()}")
print(f"[data] zero-response samples: {(V2['train_resp_lens']==0).sum()}")


def cv_eval(X, y, C=0.01, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aurocs = [], []
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xva = sc.transform(X[va])
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xva)[:, 1]
        # Tune threshold on inner split for accuracy.
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        i_tr, i_va = next(inner.split(Xtr, y[tr]))
        sc2 = StandardScaler().fit(X[tr][i_tr])
        clf2 = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=None)
        clf2.fit(sc2.transform(X[tr][i_tr]), y[tr][i_tr])
        prob_va = clf2.predict_proba(sc2.transform(X[tr][i_va]))[:, 1]
        cand = np.unique(np.concatenate([prob_va, np.linspace(0.0, 1.0, 101)]))
        best_t, best_acc = 0.5, -1.0
        for t in cand:
            acc = accuracy_score(y[tr][i_va], (prob_va >= t).astype(int))
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        pred = (prob >= best_t).astype(int)
        accs.append(accuracy_score(y[va], pred))
        aurocs.append(roc_auc_score(y[va], prob))
    return np.mean(accs), np.mean(aurocs)


def main() -> None:
    print("\nResponse-only pooling sweep (LogReg C=0.01, 5-fold CV, threshold tuned for acc):")
    print(f"  {'layer':>6} {'resp_first':>15} {'resp_mean':>15} {'resp_max':>15} {'full_last v1':>15}")
    for layer in range(n_layers):
        results = {}
        for pool in ["resp_first", "resp_mean", "resp_max"]:
            X = V2[f"train_{pool}"][:, layer, :].astype(np.float32)
            results[pool] = cv_eval(X, y)
        # v1 baseline: full-sequence last token
        Xv1 = V1["train_last"][:, layer, :].astype(np.float32)
        results["full_last"] = cv_eval(Xv1, y)
        print(
            f"  {layer:>6} "
            f"{results['resp_first'][0]*100:>6.2f}/{results['resp_first'][1]*100:>6.2f} "
            f"{results['resp_mean'][0]*100:>6.2f}/{results['resp_mean'][1]*100:>6.2f} "
            f"{results['resp_max'][0]*100:>6.2f}/{results['resp_max'][1]*100:>6.2f} "
            f"{results['full_last'][0]*100:>6.2f}/{results['full_last'][1]*100:>6.2f}"
        )

    # Best (layer, pool)
    print("\nTop 10 by accuracy across all (layer, pool) combos including v1:")
    rows = []
    for pool in ["resp_first", "resp_mean", "resp_max"]:
        for l in range(n_layers):
            X = V2[f"train_{pool}"][:, l, :].astype(np.float32)
            a, au = cv_eval(X, y)
            rows.append((pool, l, a, au))
    for l in range(n_layers):
        X = V1["train_last"][:, l, :].astype(np.float32)
        a, au = cv_eval(X, y)
        rows.append(("full_last", l, a, au))
    rows.sort(key=lambda r: -r[2])
    for pool, l, a, au in rows[:10]:
        print(f"  pool={pool:<10} layer={l:>2}  acc={a*100:.2f}  auroc={au*100:.2f}")


if __name__ == "__main__":
    main()
