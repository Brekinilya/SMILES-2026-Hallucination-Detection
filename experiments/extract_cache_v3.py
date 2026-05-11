"""
Cache for the "last K percent of tokens" heuristic.

For each sample, layer in {11..18} and fraction f in {0.20, 0.30, 0.40,
0.50, 0.70}, save max-pool and mean-pool over the last f * seq_len real
tokens. Used by sweep_heuristic.py to test whether a simple cut works as
well as the exact prompt/response boundary. It doesn't.
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
OUT_FILE = CACHE_DIR / "features_v3.npz"

LAYERS = list(range(11, 19))  # 11..18 inclusive
FRACS = [0.20, 0.30, 0.40, 0.50, 0.70]


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

    def extract(texts: list[str]) -> dict:
        n = len(texts)
        out_max = {f: np.zeros((n, len(LAYERS), hidden_dim), dtype=np.float16) for f in FRACS}
        out_mean = {f: np.zeros((n, len(LAYERS), hidden_dim), dtype=np.float16) for f in FRACS}

        for start in tqdm(range(0, n, BATCH_SIZE), desc="extract", unit="batch"):
            batch = texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch,
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
                # Real positions are [0, seq_len) for right-padding (Qwen default).
                # Last K% = last n_keep tokens.
                for fi, f in enumerate(FRACS):
                    n_keep = max(1, int(round(seq_len * f)))
                    lo = seq_len - n_keep
                    hi = seq_len
                    for li, layer_idx in enumerate(LAYERS):
                        slice_ = hidden[i, layer_idx, lo:hi, :]  # (n_keep, H)
                        out_max[f][start + i, li] = slice_.max(axis=0).astype(np.float16)
                        out_mean[f][start + i, li] = slice_.mean(axis=0).astype(np.float16)

        return out_max, out_mean

    print(f"[load] {DATA_TRAIN}")
    df = pd.read_csv(DATA_TRAIN)
    train_texts = [f"{r['prompt']}{r['response']}" for _, r in df.iterrows()]
    print(f"[train] {len(train_texts)} samples")
    t0 = time.time()
    train_max, train_mean = extract(train_texts)
    print(f"[train] extract took {time.time() - t0:.1f}s")

    print(f"[load] {DATA_TEST}")
    df_test = pd.read_csv(DATA_TEST)
    test_texts = [f"{r['prompt']}{r['response']}" for _, r in df_test.iterrows()]
    print(f"[test] {len(test_texts)} samples")
    t0 = time.time()
    test_max, test_mean = extract(test_texts)
    print(f"[test] extract took {time.time() - t0:.1f}s")

    print(f"[save] {OUT_FILE}")
    save_kwargs = {
        "layers": np.array(LAYERS, dtype=np.int32),
        "fracs": np.array(FRACS, dtype=np.float32),
    }
    for f in FRACS:
        key_f = f"f{int(f*100):02d}"  # f20, f30, f40, f50, f70
        save_kwargs[f"train_{key_f}_max"] = train_max[f]
        save_kwargs[f"train_{key_f}_mean"] = train_mean[f]
        save_kwargs[f"test_{key_f}_max"] = test_max[f]
        save_kwargs[f"test_{key_f}_mean"] = test_mean[f]
    np.savez_compressed(OUT_FILE, **save_kwargs)
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"[done] cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
