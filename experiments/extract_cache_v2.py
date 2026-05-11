"""
Like extract_cache.py but pools ONLY over response tokens.

To find the response start I tokenize the prompt separately and take its
length. When the prompt+response is too long for the 512 window and the
response gets truncated, resp_lens is 0 and the script falls back to the
full sequence for that sample.

Saves cache/features_v2.npz with resp_mean, resp_max, resp_first per
layer, plus resp_start and resp_lens per sample.
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
OUT_FILE = CACHE_DIR / "features_v2.npz"


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

    n_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size
    print(f"[model] n_layers={n_layers}  hidden_dim={hidden_dim}")

    def extract(prompts: list[str], responses: list[str]) -> dict:
        n = len(prompts)
        resp_mean = np.zeros((n, n_layers, hidden_dim), dtype=np.float16)
        resp_max = np.zeros((n, n_layers, hidden_dim), dtype=np.float16)
        resp_first = np.zeros((n, n_layers, hidden_dim), dtype=np.float16)
        resp_start = np.zeros(n, dtype=np.int32)
        resp_lens = np.zeros(n, dtype=np.int32)

        # Pre-compute prompt token lengths (same tokenization as solution.py)
        prompt_lens = []
        for p in prompts:
            ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            prompt_lens.append(len(ids))

        for start in tqdm(range(0, n, BATCH_SIZE), desc="extract", unit="batch"):
            batch_prompts = prompts[start : start + BATCH_SIZE]
            batch_responses = responses[start : start + BATCH_SIZE]
            batch_full = [p + r for p, r in zip(batch_prompts, batch_responses)]
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
                out = model(input_ids=ids, attention_mask=am)
            hidden = torch.stack(out.hidden_states, dim=1).float().cpu().numpy()
            mask_np = am.cpu().numpy().astype(bool)

            for i in range(hidden.shape[0]):
                mask = mask_np[i]
                seq_len = int(mask.sum())
                # Tokenizer pads on the right by default for AutoTokenizer with
                # left-pad disabled, so real tokens occupy positions [0, seq_len)
                # within the padded sequence.
                # Compute response start: number of prompt tokens, but capped
                # at seq_len because right-truncation may chop off the response.
                rs = min(batch_pl[i], seq_len)
                rl = max(0, seq_len - rs)

                # Position in the padded array of the first response token.
                # If padding side is "left", we'd need to shift; verify:
                # AutoTokenizer for Qwen defaults to "right" padding.
                padding_side = tokenizer.padding_side
                if padding_side == "right":
                    # Real tokens at positions [0, seq_len)
                    resp_pos_lo = rs
                    resp_pos_hi = seq_len  # exclusive
                else:
                    pad_len = hidden.shape[2] - seq_len
                    resp_pos_lo = pad_len + rs
                    resp_pos_hi = pad_len + seq_len

                resp_start[start + i] = resp_pos_lo
                resp_lens[start + i] = rl

                if rl == 0:
                    # Response was completely truncated - fall back to the
                    # whole real sequence to avoid empty pools.
                    if padding_side == "right":
                        resp_pos_lo, resp_pos_hi = 0, seq_len
                    else:
                        pad_len = hidden.shape[2] - seq_len
                        resp_pos_lo, resp_pos_hi = pad_len, pad_len + seq_len

                resp_slice = hidden[i, :, resp_pos_lo:resp_pos_hi, :]  # (L, R, H)
                resp_mean[start + i] = resp_slice.mean(axis=1).astype(np.float16)
                resp_max[start + i] = resp_slice.max(axis=1).astype(np.float16)
                resp_first[start + i] = resp_slice[:, 0, :].astype(np.float16)

        return dict(
            resp_mean=resp_mean,
            resp_max=resp_max,
            resp_first=resp_first,
            resp_start=resp_start,
            resp_lens=resp_lens,
        )

    print(f"[load] {DATA_TRAIN}")
    df = pd.read_csv(DATA_TRAIN)
    train_prompts = df["prompt"].tolist()
    train_responses = df["response"].tolist()
    print(f"[train] {len(train_prompts)} samples")
    t0 = time.time()
    train = extract(train_prompts, train_responses)
    print(f"[train] extract took {time.time() - t0:.1f}s  "
          f"resp_lens: min={train['resp_lens'].min()}  "
          f"median={int(np.median(train['resp_lens']))}  "
          f"max={train['resp_lens'].max()}  "
          f"zero={(train['resp_lens']==0).sum()}")

    print(f"[load] {DATA_TEST}")
    df_test = pd.read_csv(DATA_TEST)
    test_prompts = df_test["prompt"].tolist()
    test_responses = df_test["response"].tolist()
    print(f"[test] {len(test_prompts)} samples")
    t0 = time.time()
    test = extract(test_prompts, test_responses)
    print(f"[test] extract took {time.time() - t0:.1f}s  "
          f"resp_lens: min={test['resp_lens'].min()}  "
          f"median={int(np.median(test['resp_lens']))}  "
          f"max={test['resp_lens'].max()}  "
          f"zero={(test['resp_lens']==0).sum()}")

    print(f"[save] {OUT_FILE}")
    np.savez_compressed(
        OUT_FILE,
        train_resp_mean=train["resp_mean"],
        train_resp_max=train["resp_max"],
        train_resp_first=train["resp_first"],
        train_resp_start=train["resp_start"],
        train_resp_lens=train["resp_lens"],
        test_resp_mean=test["resp_mean"],
        test_resp_max=test["resp_max"],
        test_resp_first=test["resp_first"],
        test_resp_start=test["resp_start"],
        test_resp_lens=test["resp_lens"],
    )
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"[done] cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
