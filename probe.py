"""
probe.py: HallucinationProbe classifier.

A small ensemble of 5 logistic regressions trained on bootstrap samples of
the training data. Predictions are the averaged class-1 probability across
the 5 fits. The decision threshold is picked inside fit() from 5-fold
out-of-fold probabilities so that the final probe trained in solution.py
ships with a non-default threshold.

A few design choices:

- C is fixed at 0.01 (strong L2). Stronger regularization beats weaker on
  this 689 x 1792 problem. I checked C in {0.001, 0.01, 0.05, 0.1, 0.5,
  1.0, 10.0}; 0.01 to 0.05 was the best range. Adaptive C-search inside
  fit() was within +/-0.02 acc of fixed 0.01, so I kept it simple.
- class_weight=None, not 'balanced'. Train prior is ~70/30 hallucinated
  to truthful. Balancing flattens calibration and cost about 1 point of
  accuracy on every layer I tested.
- 5 bootstrap fits with fixed seeds. Bootstrap aggregation is a simple
  way to reduce variance on a small dataset, and each fit is cheap.
- Threshold tuned for accuracy (the competition primary metric), not for
  F1. F1 tuning predicts more positives and accuracy drops by about 1
  point.
- Threshold is picked from 5-fold OOF predictions inside fit() so that
  every training sample contributes once. evaluate.py later overwrites
  the threshold per-fold via fit_hyperparameters.

The class still subclasses nn.Module to satisfy the evaluator contract.
forward() returns logits from a single nn.Linear that mirrors the
average of the ensemble weights; the evaluator itself uses predict() and
predict_proba() directly, so this is just for shape correctness.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Hyperparameters picked in experiments/final_search.py and
# experiments/response_ensemble.py on 5-fold stratified CV.
LOGREG_C: float = 0.01
RANDOM_STATE: int = 42
INNER_FOLDS: int = 5     # k-fold inside fit() for OOF threshold tuning
N_BOOTSTRAP: int = 5     # number of bootstrap fits in the production ensemble
BOOTSTRAP_SEEDS: tuple[int, ...] = (0, 1, 7, 42, 123)


class HallucinationProbe(nn.Module):
    """Binary classifier for hallucination detection on hidden-state features.

    Stores N_BOOTSTRAP (StandardScaler, LogReg) pipelines fit on bootstrap
    samples. predict_proba averages the class-1 probability over them.
    forward() uses a single nn.Linear that mirrors the average ensemble
    weights, only to satisfy the nn.Module contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scalers: list[StandardScaler] = []
        self._clfs: list[LogisticRegression] = []
        self._net: nn.Linear | None = None
        self._threshold: float = 0.5

    # Internal helpers ----------------------------------------------------
    def _fit_one(self, X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(
            C=LOGREG_C, max_iter=3000, solver="lbfgs", class_weight=None,
        )
        clf.fit(sc.transform(X), y.astype(int))
        return sc, clf

    def _fit_bootstrap_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit N_BOOTSTRAP independent LogReg pipelines on bootstrap samples
        of (X, y) and store them all for averaged inference."""
        self._scalers = []
        self._clfs = []
        n = len(y)
        for seed in BOOTSTRAP_SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, size=n, replace=True)
            sc, clf = self._fit_one(X[idx], y[idx])
            self._scalers.append(sc)
            self._clfs.append(clf)
        self._mirror_to_torch_avg()

    def _mirror_to_torch_avg(self) -> None:
        """Put the averaged ensemble weights into a single nn.Linear so
        forward() returns logits in roughly the same regime as
        predict_proba. The evaluator does not call forward() in practice,
        this is only here to satisfy the nn.Module contract.
        """
        if not self._clfs:
            return
        ws = np.stack([c.coef_[0] for c in self._clfs], axis=0).astype(np.float32)
        bs = np.array([float(c.intercept_[0]) for c in self._clfs], dtype=np.float32)
        w = ws.mean(axis=0)
        b = float(bs.mean())
        layer = nn.Linear(w.shape[0], 1)
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(w).unsqueeze(0))
            layer.bias.copy_(torch.tensor([b], dtype=torch.float32))
        self._net = layer

    @staticmethod
    def _best_accuracy_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        cand = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_acc = 0.5, -1.0
        for t in cand:
            acc = accuracy_score(y_true, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        return best_t

    # Public API ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit the bootstrap ensemble and pick the accuracy-optimal threshold.

        Threshold selection is done first via 5-fold OOF on the input
        (X, y) using a single LogReg per fold. Then the production
        ensemble is fit on the full input.
        """
        y_int = y.astype(int)
        n = len(y_int)

        oof_probs = np.full(n, np.nan, dtype=np.float64)
        try:
            skf = StratifiedKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE
            )
            for tr_idx, va_idx in skf.split(np.arange(n), y_int):
                # A single LogReg is enough for OOF threshold selection;
                # the production ensemble lives below.
                sc, clf = self._fit_one(X[tr_idx], y_int[tr_idx])
                oof_probs[va_idx] = clf.predict_proba(sc.transform(X[va_idx]))[:, 1]
            self._threshold = self._best_accuracy_threshold(oof_probs, y_int)
        except ValueError:
            self._threshold = 0.5

        self._fit_bootstrap_ensemble(X, y_int)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Re-tune the decision threshold on a held-out validation set."""
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_accuracy_threshold(probs, y_val.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average class probabilities across the bootstrap ensemble."""
        if not self._clfs:
            raise RuntimeError(
                "Probe has not been fitted yet. Call fit() before predict_proba()."
            )
        probs_pos = np.zeros(len(X), dtype=np.float64)
        for sc, clf in zip(self._scalers, self._clfs):
            probs_pos += clf.predict_proba(sc.transform(X))[:, 1]
        probs_pos /= len(self._clfs)
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)
