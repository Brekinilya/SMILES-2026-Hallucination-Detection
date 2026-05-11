"""
splitting.py: train/val/test splits for the evaluation harness.

5-fold StratifiedKFold. For each outer fold the held-out chunk becomes
idx_test. The remaining 4/5 is split once more (stratified, 80/20) into
idx_train and idx_val, and idx_val is used by probe.fit_hyperparameters
to tune the per-fold threshold.

Why k-fold and not one random split: the dataset is small (689 labelled
samples), so a single split is noisy. I measured per-seed std around 1.0
accuracy on one split vs about 0.3 across 5 folds. Averaging 5 folds gives
a more honest estimate of how the probe behaves.

I do not use GroupKFold because all 689 prompts are unique (I checked at
the start), so there is no leakage to worry about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

N_FOLDS = 5
INNER_VAL_FRAC = 0.2  # fraction of (train+val) reserved for validation


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return N_FOLDS stratified k-fold splits with an inner validation
    split per fold for threshold tuning.

    Args:
        y:            label array of shape (N,) with values in {0, 1}.
        df:           optional DataFrame, unused here.
        test_size:    ignored (folds are equal at 1/N_FOLDS).
        val_size:     ignored (inner split uses INNER_VAL_FRAC).
        random_state: seed for reproducible folds.

    Returns:
        list of length N_FOLDS; each element is
        (idx_train, idx_val, idx_test) of integer indices.
    """
    del test_size, val_size

    idx_all = np.arange(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(skf.split(idx_all, y)):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=INNER_VAL_FRAC,
            random_state=random_state + fold_idx,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
