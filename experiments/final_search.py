"""
Try several variations around the iter-1 winner (L14 last token,
LogReg(C=0.01), class_weight=None, threshold for accuracy):

  - MLP with same regime
  - LogReg + MLP probability averaging
  - L13+L14 concat
  - L14 last+mean pool concat
  - PCA before LogReg
  - 7 seeds for variance
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = np.load(REPO_ROOT / "cache" / "features.npz")
y = CACHE["train_labels"].astype(int)
N = len(y)
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[data] N={N}  device={device}")


def best_threshold(probs, y_true, metric="acc"):
    cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_v = 0.5, -1.0
    for t in cand:
        pred = (probs >= t).astype(int)
        v = accuracy_score(y_true, pred) if metric == "acc" else f1_score(y_true, pred, zero_division=0)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def cv_eval(get_train_proba_test_proba, X, y, n_folds=5, seed=42, threshold_metric="acc"):
    """get_train_proba_test_proba(X_tr, y_tr, X_va, y_va, X_te) -> (p_va, p_te)"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, f1s, aurocs = [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        # inner split for threshold
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        i_tr, i_va = next(inner.split(X[tr_idx], y[tr_idx]))
        Xtr_in = X[tr_idx][i_tr]; ytr_in = y[tr_idx][i_tr]
        Xva_in = X[tr_idx][i_va]; yva_in = y[tr_idx][i_va]
        # tune threshold on inner-val
        p_va, _ = get_train_proba_test_proba(Xtr_in, ytr_in, Xva_in, yva_in, X[te_idx])
        t = best_threshold(p_va, yva_in, metric=threshold_metric)
        # refit on full outer train
        _, p_te = get_train_proba_test_proba(X[tr_idx], y[tr_idx], None, None, X[te_idx])
        pred = (p_te >= t).astype(int)
        accs.append(accuracy_score(y[te_idx], pred))
        f1s.append(f1_score(y[te_idx], pred, zero_division=0))
        aurocs.append(roc_auc_score(y[te_idx], p_te))
    return np.mean(accs), np.mean(f1s), np.mean(aurocs)


def logreg_factory(C=0.01, class_weight=None):
    def f(Xtr, ytr, Xva, yva, Xte):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight=class_weight,
                                 solver="lbfgs").fit(sc.transform(Xtr), ytr)
        p_va = clf.predict_proba(sc.transform(Xva))[:, 1] if Xva is not None else None
        p_te = clf.predict_proba(sc.transform(Xte))[:, 1]
        return p_va, p_te
    return f


def mlp_factory(hidden=128, dropout=0.3, lr=1e-3, wd=1e-3, epochs=120, seed=SEED):
    def f(Xtr, ytr, Xva, yva, Xte):
        torch.manual_seed(seed)
        sc = StandardScaler().fit(Xtr)
        Xtr_s = torch.from_numpy(sc.transform(Xtr).astype(np.float32)).to(device)
        Xte_s = torch.from_numpy(sc.transform(Xte).astype(np.float32)).to(device)
        ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
        net = nn.Sequential(
            nn.Linear(Xtr.shape[1], hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        crit = nn.BCEWithLogitsLoss()  # no pos_weight
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(net(Xtr_s).squeeze(-1), ytr_t)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(net(Xte_s).squeeze(-1)).cpu().numpy()
            p_va = None
            if Xva is not None:
                Xva_s = torch.from_numpy(sc.transform(Xva).astype(np.float32)).to(device)
                p_va = torch.sigmoid(net(Xva_s).squeeze(-1)).cpu().numpy()
        return p_va, p_te
    return f


def stack_factory(C=0.01, mlp_hidden=128, mlp_epochs=120):
    """Average LogReg and MLP probabilities."""
    lr_f = logreg_factory(C=C, class_weight=None)
    mlp_f = mlp_factory(hidden=mlp_hidden, epochs=mlp_epochs)
    def f(Xtr, ytr, Xva, yva, Xte):
        p_va_lr, p_te_lr = lr_f(Xtr, ytr, Xva, yva, Xte)
        p_va_mlp, p_te_mlp = mlp_f(Xtr, ytr, Xva, yva, Xte)
        p_va = None if p_va_lr is None else (p_va_lr + p_va_mlp) / 2.0
        p_te = (p_te_lr + p_te_mlp) / 2.0
        return p_va, p_te
    return f


def make_X(spec: str) -> np.ndarray:
    """Spec mini-DSL: 'last:14', 'last:13,14', 'last:14|mean:14', 'pca64:last:14'"""
    parts = spec.split("|")
    arrs = []
    for part in parts:
        if part.startswith("pca"):
            # e.g. pca64:last:14
            n_comp_part, sub = part.split(":", 1)
            n_comp = int(n_comp_part[3:])
            X = make_X(sub)
            X = StandardScaler().fit_transform(X)
            X = PCA(n_components=n_comp, random_state=SEED).fit_transform(X)
            arrs.append(X.astype(np.float32))
        else:
            pool, layers_str = part.split(":")
            layers = [int(x) for x in layers_str.split(",")]
            X = np.concatenate(
                [CACHE[f"train_{pool}"][:, l, :] for l in layers], axis=1
            ).astype(np.float32)
            arrs.append(X)
    return np.concatenate(arrs, axis=1)


def main() -> None:
    EXPERIMENTS = [
        ("logreg  L14 last     C=0.01", "last:14",        logreg_factory(C=0.01)),
        ("logreg  L14 last     C=0.05", "last:14",        logreg_factory(C=0.05)),
        ("logreg  L13+14 last  C=0.01", "last:13,14",     logreg_factory(C=0.01)),
        ("logreg  L14 last+mean        ", "last:14|mean:14", logreg_factory(C=0.01)),
        ("logreg  pca64 L14 last      ", "pca64:last:14",  logreg_factory(C=1.0)),
        ("logreg  pca128 L14 last     ", "pca128:last:14", logreg_factory(C=1.0)),
        ("mlp     L14 last  h=64       ", "last:14",       mlp_factory(hidden=64,  epochs=80)),
        ("mlp     L14 last  h=128      ", "last:14",       mlp_factory(hidden=128, epochs=120)),
        ("mlp     L14 last  h=256      ", "last:14",       mlp_factory(hidden=256, epochs=120)),
        ("stack   L14 last  lr+mlp     ", "last:14",       stack_factory()),
    ]

    print(f"\n{'experiment':<32} {'acc':>7} {'f1':>7} {'auroc':>7}")
    print("-" * 60)
    for name, spec, fac in EXPERIMENTS:
        X = make_X(spec)
        a, f, au = cv_eval(fac, X, y, threshold_metric="acc")
        print(f"  {name:<30} {a*100:6.2f}% {f*100:6.2f}% {au*100:6.2f}%")

    # Also: 7-seed stability for the chosen config
    print("\n=== Stability: logreg L14 C=0.01 cw=None across 7 seeds ===")
    accs, aurocs = [], []
    for seed in [0, 1, 7, 42, 123, 2024, 31415]:
        a, f, au = cv_eval(logreg_factory(C=0.01), make_X("last:14"), y,
                           seed=seed, threshold_metric="acc")
        accs.append(a); aurocs.append(au)
        print(f"  seed={seed:6d}  acc={a*100:.2f}  auroc={au*100:.2f}")
    print(f"  mean: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}  "
          f"auroc={np.mean(aurocs)*100:.2f}±{np.std(aurocs)*100:.2f}")


if __name__ == "__main__":
    main()
