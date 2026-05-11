"""
Run Qwen2.5-0.5B forward over train and test once and save per-layer
aggregates so the rest of the experiments do not need the LLM.

Saves cache/features.npz with last / mean / max pool over the full
(prompt+response) sequence for every transformer layer, plus the sample
labels and sequence lengths.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
BATCH_SIZE = 8
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
DATA_TRAIN = os.path.join(REPO_ROOT, "data", "dataset.csv")
DATA_TEST = os.path.join(REPO_ROOT, "data", "test.csv")
CACHE_FILE = os.path.join(CACHE_DIR, "features.npz")


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
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

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding output
    hidden_dim = model.config.hidden_size
    print(f"[model] n_layers={n_layers}  hidden_dim={hidden_dim}")

    def extract(texts: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        n = len(texts)
        feats = {
            "last": np.zeros((n, n_layers, hidden_dim), dtype=np.float16),
            "mean": np.zeros((n, n_layers, hidden_dim), dtype=np.float16),
            "max": np.zeros((n, n_layers, hidden_dim), dtype=np.float16),
        }
        seq_lens = np.zeros(n, dtype=np.int32)

        for start in tqdm(
            range(0, n, BATCH_SIZE),
            desc="extract",
            unit="batch",
        ):
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
                out = model(input_ids=ids, attention_mask=am)
            # (B, L, T, H) in fp32 on cpu
            hidden = (
                torch.stack(out.hidden_states, dim=1).float().cpu().numpy()
            )
            mask_np = am.cpu().numpy().astype(bool)

            for i in range(hidden.shape[0]):
                mask = mask_np[i]
                seq_len = int(mask.sum())
                seq_lens[start + i] = seq_len
                h = hidden[i][:, mask, :]  # (L, seq_len, H)
                # last real token (= response end token, typically <|endoftext|>)
                feats["last"][start + i] = h[:, -1, :].astype(np.float16)
                feats["mean"][start + i] = h.mean(axis=1).astype(np.float16)
                feats["max"][start + i] = h.max(axis=1).astype(np.float16)

        return feats, seq_lens

    # Train ------------------------------------------------------------------
    print(f"[load] {DATA_TRAIN}")
    df = pd.read_csv(DATA_TRAIN)
    train_texts = [f"{r['prompt']}{r['response']}" for _, r in df.iterrows()]
    train_labels = df["label"].astype(int).to_numpy()
    print(f"[train] {len(train_texts)} samples  "
          f"({int(train_labels.sum())} hallucinated / "
          f"{int((train_labels == 0).sum())} truthful)")

    t0 = time.time()
    train_feats, train_lens = extract(train_texts)
    print(f"[train] extract took {time.time() - t0:.1f}s")

    # Test -------------------------------------------------------------------
    print(f"[load] {DATA_TEST}")
    df_test = pd.read_csv(DATA_TEST)
    test_texts = [f"{r['prompt']}{r['response']}" for _, r in df_test.iterrows()]
    print(f"[test] {len(test_texts)} samples")

    t0 = time.time()
    test_feats, test_lens = extract(test_texts)
    print(f"[test] extract took {time.time() - t0:.1f}s")

    # Save -------------------------------------------------------------------
    print(f"[save] {CACHE_FILE}")
    np.savez_compressed(
        CACHE_FILE,
        train_last=train_feats["last"],
        train_mean=train_feats["mean"],
        train_max=train_feats["max"],
        train_seq_lens=train_lens,
        train_labels=train_labels,
        test_last=test_feats["last"],
        test_mean=test_feats["mean"],
        test_max=test_feats["max"],
        test_seq_lens=test_lens,
    )
    size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    print(f"[done] cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
