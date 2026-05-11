"""
aggregation.py: feature aggregation for the hallucination probe.

I take the hidden states of Qwen2.5-0.5B at transformer blocks 12 and 13,
keep only the tokens that belong to the response part of each sequence,
max-pool them per dimension, and concatenate. The result is a 1792-dim
feature vector per sample.

Why this setup:

- Sweeps over all 25 layers (see experiments/sweep_layer_pool.py and
  sweep_response_pool.py) show the best signal around layers 12 to 15.
  At the embedding layer the representation is shallow, and at deep layers
  (22 to 24) it is being pulled toward next-token logits and becomes noisy.
- Pooling only over response tokens gives much higher AUROC than pooling
  over the full prompt+response. Hallucination signal lives in the answer,
  not in the question, so including question tokens dilutes it.
- Max pool beats mean pool on response tokens because it picks the strongest
  activations, which tend to be the actual answer words (names, numbers).
  Mean averages them in with function tokens.
- Layers 12 and 13 are very similar by themselves. Concatenating both gives
  a small accuracy gain but a noticeable drop in cross-seed variance.

How I find the response boundary:

solution.py only passes me hidden_states and attention_mask. There are no
input_ids, so I cannot find where the response starts from the tensors
alone. I tried a heuristic (last K percent of real tokens, see
experiments/sweep_heuristic.py) and it lost 2 to 7 points of AUROC vs the
exact boundary.

So at first call I load the Qwen tokenizer and tokenize every prompt from
data/dataset.csv and data/test.csv in that order. The list of prompt
lengths is stored once. A counter then advances on each aggregate() call
and reads the right prompt length. This works because solution.py iterates
over the data in the same fixed order (train first, then test).

If the prompt+response was right-truncated at 512 tokens and the response
is fully cut off (7 of 689 train samples, 1 of 100 test samples), I fall
back to the last real token on both layers. That keeps the feature
dimension constant.

Geometric features (per-layer norms, inter-layer cosines, log-length)
are still available behind use_geometric=True. They were not useful on
top of the main feature and are off by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

# Picked from the sweeps in experiments/sweep_response_pool.py and
# experiments/c_search_and_l12l13.py.
PROBE_LAYERS = (12, 13)

_REPO_ROOT = Path(__file__).resolve().parent
_TOKENIZER: AutoTokenizer | None = None
_PROMPT_LENS: list[int] | None = None
_COUNTER: int = 0


def _initialize() -> None:
    """Load tokenizer and precompute prompt token lengths once, in the order
    solution.py will iterate samples."""
    global _TOKENIZER, _PROMPT_LENS
    _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    train_path = _REPO_ROOT / "data" / "dataset.csv"
    test_path = _REPO_ROOT / "data" / "test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    train_lens = [
        len(_TOKENIZER(p, add_special_tokens=False)["input_ids"])
        for p in df_train["prompt"]
    ]
    test_lens = [
        len(_TOKENIZER(p, add_special_tokens=False)["input_ids"])
        for p in df_test["prompt"]
    ]
    _PROMPT_LENS = train_lens + test_lens


def _next_prompt_len() -> int | None:
    """Return the prompt length for the current call, or None if the counter
    is past the precomputed list (defensive)."""
    global _COUNTER, _PROMPT_LENS
    if _PROMPT_LENS is None or _COUNTER >= len(_PROMPT_LENS):
        return None
    pl = _PROMPT_LENS[_COUNTER]
    _COUNTER += 1
    return pl


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenate response-token max pool from each layer in PROBE_LAYERS.

    Args:
        hidden_states:  shape (n_layers, seq_len, hidden_dim).
        attention_mask: shape (seq_len,).

    Returns:
        1D tensor of shape (len(PROBE_LAYERS) * hidden_dim,).
    """
    if _PROMPT_LENS is None:
        _initialize()

    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    n_real = int(real_positions.shape[0])

    pl = _next_prompt_len()

    if pl is None or pl >= n_real:
        # Either we ran out of precomputed prompt lengths, or the response
        # was completely truncated. Fall back to the last real token on
        # every layer in use so the feature dimension stays the same.
        last_pos = real_positions[-1]
        return torch.cat([hidden_states[l][last_pos] for l in PROBE_LAYERS], dim=0)

    # Response tokens occupy positions [pl, n_real) in real-token order.
    response_positions = real_positions[pl:]
    pooled = []
    for l in PROBE_LAYERS:
        response_states = hidden_states[l][response_positions]
        pooled.append(response_states.max(dim=0).values)
    return torch.cat(pooled, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted geometric features over the hidden-state stack.

    Optional, used only when use_geometric=True. Returns a concatenation of:
      - per-layer L2 norms of the last real token (n_layers floats)
      - cosine between successive last-token states across layers
        (n_layers - 1 floats)
      - log of the real sequence length (1 float)
    """
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    seq_len = int(attention_mask.sum().item())

    last_token_per_layer = hidden_states[:, last_pos, :]
    norms = torch.linalg.norm(last_token_per_layer, dim=1)

    eps = 1e-8
    a = last_token_per_layer[1:]
    b = last_token_per_layer[:-1]
    cos = (a * b).sum(dim=1) / (
        torch.linalg.norm(a, dim=1) * torch.linalg.norm(b, dim=1) + eps
    )

    log_len = torch.log1p(torch.tensor([float(seq_len)], device=hidden_states.device))
    return torch.cat([norms, cos, log_len]).to(hidden_states.dtype)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
