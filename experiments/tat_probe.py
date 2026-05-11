"""
Truth-as-a-Trajectory probe (Damirchi et al. 2026).

For each sample, build the sequence of inter-layer displacement vectors
delta_l = h_l - h_{l-1} and feed it to a small LSTM. The classifier head
runs on the LSTM's final hidden state. Idea is that valid generations
have smooth layer-to-layer changes and hallucinations have jumps.

Did not beat single-layer LogReg on this dataset (689 samples is too few
for an LSTM to train well).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = np.load(REPO_ROOT / "cache" / "features.npz")
N_FOLDS = 5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_displacements(arr: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    arr  : (N, L, H)  hidden states
    out  : (N, L-1, H) displacement vectors (h_l - h_{l-1})
    if normalize: each delta is L2-normalized (focus on direction).
    """
    delta = arr[:, 1:, :] - arr[:, :-1, :]
    if normalize:
        norm = np.linalg.norm(delta, axis=2, keepdims=True) + 1e-8
        delta = delta / norm
    return delta.astype(np.float32)


class TaTProbe(nn.Module):
    def __init__(self, hidden_dim: int, lstm_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            num_layers=1,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) where T = n_layers - 1
        out, (h_n, _) = self.lstm(x)
        # use the final hidden state
        z = h_n[-1]                 # (B, lstm_hidden)
        return self.head(z).squeeze(-1)  # (B,)


def train_tat(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    epochs: int = 60, lr: float = 1e-3, wd: float = 1e-3, batch_size: int = 32,
    lstm_hidden: int = 64, seed: int = SEED,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    net = TaTProbe(hidden_dim=X_tr.shape[2], lstm_hidden=lstm_hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr = torch.from_numpy(X_tr).float().to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)

    n = len(X_tr)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad()
            logits = net(Xtr[idx])
            loss = crit(logits, ytr[idx])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        Xva = torch.from_numpy(X_va).float().to(device)
        prob = torch.sigmoid(net(Xva)).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return accuracy_score(y_va, pred), roc_auc_score(y_va, prob)


def main() -> None:
    y = CACHE["train_labels"].astype(int)
    n = len(y)
    print(f"[data] N={n}  pos={int(y.sum())}  neg={int((y==0).sum())}  device={device}")

    for pool in ["last", "mean"]:
        print(f"\n[exp] TaT  pool={pool}  normalize=True")
        H = CACHE[f"train_{pool}"].astype(np.float32)
        delta = build_displacements(H, normalize=True)
        print(f"       trajectory shape: {delta.shape}  (N, L-1, H)")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        accs, aurocs = [], []
        for fold, (tr, va) in enumerate(skf.split(np.arange(n), y)):
            a, au = train_tat(delta[tr], y[tr], delta[va], y[va])
            accs.append(a); aurocs.append(au)
            print(f"  fold {fold+1}: acc={a*100:5.2f}  auroc={au*100:5.2f}")
        print(f"  mean   : acc={np.mean(accs)*100:5.2f}  auroc={np.mean(aurocs)*100:5.2f}")

    print("\n[exp] TaT  pool=last  normalize=False (raw deltas)")
    H = CACHE["train_last"].astype(np.float32)
    delta = build_displacements(H, normalize=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs, aurocs = [], []
    for fold, (tr, va) in enumerate(skf.split(np.arange(n), y)):
        a, au = train_tat(delta[tr], y[tr], delta[va], y[va])
        accs.append(a); aurocs.append(au)
        print(f"  fold {fold+1}: acc={a*100:5.2f}  auroc={au*100:5.2f}")
    print(f"  mean   : acc={np.mean(accs)*100:5.2f}  auroc={np.mean(aurocs)*100:5.2f}")


if __name__ == "__main__":
    main()
