"""
Cache for top-K max-pool over response tokens, per dimension.

For each sample, layer in {11..15} and K in {1, 3, 5, 10, 20}, save the
mean of the top-K largest activations among response tokens per dim.
K=1 is plain max, K -> n_response gives mean. Used by sweep_topk.py and
topk_stability.py to test whether top-K is more robust than top-1.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
BATCH_SIZE = 8
CACHE_DIR = REPO_ROOT / "cache"
DATA_TRAIN = REPO_ROOT / "data" / "dataset.csv"
DATA_TEST = REPO_ROOT / "data" / "test.csv"
OUT_FILE = CACHE_DIR / "features_v4.npz"

LAYERS = [11, 12, 13, 14, 15]
KS = [1, 3, 5, 10, 20]


def topk_pool(slice_lh: np.ndarray, k: int) -> np.ndarray:
    """slice_lh: (n_response_tokens, hidden_dim). Returns (hidden_dim,)
    where each entry is the mean of the top-K values along the token axis.
    Falls back to mean if there are fewer than K real tokens."""
    n_resp = slice_lh.shape[0]
    if n_resp == 0:
        return np.zeros(slice_lh.shape[1], dtype=np.float32)
    k_eff = min(k, n_resp)
    # Partition is faster than full sort for top-K.
    if k_eff == n_resp:
        return slice_lh.mean(axis=0)
    # axis=0 sort: take last k_eff (the largest)
    part = np.partition(slice_lh, n_resp - k_eff, axis=0)[n_resp - k_eff :]
    return part.mean(axis=0)


def main() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print(f"[load] {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    hidden_dim = model.config.hidden_size

    def extract(prompts: list[str], responses: list[str]) -> dict:
        n = len(prompts)
        # storage[k][layer_pos] -> (n, hidden_dim)
        storage = {k: np.zeros((n, len(LAYERS), hidden_dim), dtype=np.float16) for k in KS}

        prompt_lens = [
            len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
        ]

        for start in tqdm(range(0, n, BATCH_SIZE), desc="extract", unit="batch"):
            batch_full = [
                p + r for p, r in zip(
                    prompts[start : start + BATCH_SIZE],
                    responses[start : start + BATCH_SIZE],
                )
            ]
            batch_pl = prompt_lens[start : start + BATCH_SIZE]

            enc = tokenizer(
                batch_full,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            with torch.no_grad():
                outs = model(input_ids=ids, attention_mask=am)
            hidden = torch.stack(outs.hidden_states, dim=1).float().cpu().numpy()
            mask_np = am.cpu().numpy().astype(bool)

            for i in range(hidden.shape[0]):
                mask = mask_np[i]
                seq_len = int(mask.sum())
                rs = min(batch_pl[i], seq_len)
                lo = rs
                hi = seq_len
                if lo >= hi:
                    # Response truncated; fall back to last token (n_resp=1)
                    lo = max(0, seq_len - 1)
                    hi = seq_len

                for li, layer_idx in enumerate(LAYERS):
                    slice_lh = hidden[i, layer_idx, lo:hi, :]  # (n_response, H)
                    for k in KS:
                        v = topk_pool(slice_lh, k)
                        storage[k][start + i, li] = v.astype(np.float16)

        return storage

    print(f"[load] {DATA_TRAIN}")
    df = pd.read_csv(DATA_TRAIN)
    train_p = df["prompt"].tolist()
    train_r = df["response"].tolist()
    print(f"[train] {len(train_p)} samples")
    t0 = time.time()
    train_st = extract(train_p, train_r)
    print(f"[train] extract took {time.time() - t0:.1f}s")

    print(f"[load] {DATA_TEST}")
    df_test = pd.read_csv(DATA_TEST)
    test_p = df_test["prompt"].tolist()
    test_r = df_test["response"].tolist()
    print(f"[test] {len(test_p)} samples")
    t0 = time.time()
    test_st = extract(test_p, test_r)
    print(f"[test] extract took {time.time() - t0:.1f}s")

    print(f"[save] {OUT_FILE}")
    save = {"layers": np.array(LAYERS, dtype=np.int32), "ks": np.array(KS, dtype=np.int32)}
    for k in KS:
        save[f"train_topk{k}"] = train_st[k]
        save[f"test_topk{k}"] = test_st[k]
    np.savez_compressed(OUT_FILE, **save)
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"[done] cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
