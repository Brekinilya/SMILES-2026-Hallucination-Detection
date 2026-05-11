# SMILES-2026 Hallucination Detection: Solution Report

## Summary

I extract hidden states from Qwen2.5-0.5B for each `prompt + response` pair,
take max over response tokens at transformer layers 12 and 13, concatenate
into 1792-dim vector, and feed it to a small ensemble of 5 logistic
regressions with strong L2. Decision threshold is tuned for accuracy on
out-of-fold predictions.

## Result

5-fold stratified CV on the labelled `dataset.csv` (689 samples):

| Setup | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority baseline (predict 1 for all) | 70.10% | 82.42% | n/a |
| Default skeleton from the repo | ~62% | n/a | ~67% |
| My iter 1 (last token, layer 14, LogReg) | 71.99% | 81.12% | 70.74% |
| My iter 2 (response max-pool, layer 13, LogReg + bootstrap) | 75.32% | 83.63% | 78.74% |
| My final (response max-pool, layers 12 and 13 concat, same probe) | **75.33%** | **83.38%** | **78.92%** |

## What I changed in the code

I edited three files that the README marks as student-editable:

* `aggregation.py`: now does response-only max pool at layers 12 and 13 and
  concatenates them. To find where response starts in the token sequence,
  the module pre-tokenizes prompts from `data/dataset.csv` and `data/test.csv`
  at first call and uses a counter on subsequent calls.
* `probe.py`: replaced the MLP with 5 logistic regressions fit on bootstrap
  samples of the training data. C is fixed at 0.01, class weights are off,
  threshold is picked on out-of-fold predictions to maximize accuracy.
* `splitting.py`: 5-fold StratifiedKFold instead of one random split, with
  an inner 80/20 stratified split per fold so that `evaluate.py` can call
  `fit_hyperparameters` with a real validation set.

I did not change `model.py`, `evaluate.py` or `solution.py`.

## Method

### Aggregation

For every sample I look at the hidden states at the output of transformer
block 12 and transformer block 13 (the model has 24 blocks total). On each
of these two layers I take only the tokens that belong to the response
part of the sequence, max-pool them per dimension, and concatenate. The
resulting feature has 1792 entries.

Some context on why these choices:

* Sweeps in `experiments/sweep_layer_pool.py` and
  `experiments/sweep_response_pool.py` show a clear arc across layers. At
  the embedding (layer 0) the signal is weak, at deep layers (22-24) it
  is also weak because the representation is being pulled towards
  next-token logits. The peak is in the middle, around layers 12-15. This
  matches what the literature on linear probes of small Llama/Qwen
  models reports (Marks and Tegmark 2023, Damirchi et al. 2026).
* Pooling over only the response tokens, not the whole sequence, gives
  much higher AUROC at the same accuracy. On layer 13 with the same
  probe, the AUROC went from 71.41 (full sequence last token) to 78.55
  (response only max). The intuition: hallucination lives in the answer,
  not in the question, so reading the question parts dilutes the signal.
* Max pool was better than mean pool for response tokens. Mean averages
  in a lot of stop words and short function tokens, max picks the
  strongest activations which often correspond to the actual answer
  tokens like names or numbers.
* Layers 12 and 13 by themselves are very similar. Concatenating both
  did not give much accuracy but reduced cross-seed standard deviation
  from 1.30 to 1.14, and on the production run it also brought the test
  predicted-positive rate from 88% down to 76%, much closer to the
  training prior of 70%. So the concat acts more like a stabilizer than
  a real new signal.

### How I get the prompt/response boundary

`solution.py` does not give the aggregation function the token ids. It
gives only `hidden_states` and `attention_mask`. So I cannot find the
border between prompt and response from these tensors alone.

I tried a heuristic in `experiments/sweep_heuristic.py`: just take the
last K percent of real tokens, hoping that K around 0.3 will cover the
response. It works but loses 2 to 7 points of AUROC compared to the
exact boundary, so the cost of getting the exact answer is worth it.

The exact boundary needs the prompt length in tokens, which is not in
the inputs. I load the tokenizer once inside `aggregation.py` and
pre-tokenize every prompt from `data/dataset.csv` and `data/test.csv`,
in that order. This is the same order in which `solution.py` will call
the aggregator. A counter increments on each call and reads the prompt
length from the precomputed list.

If the prompt was so long that the full prompt+response did not fit in
the 512-token window and the response was completely truncated, the
counter check sees `pl >= n_real` and falls back to taking the last
real token (so the output dimension stays the same and the probe still
gets a feature). This happens for 7 of 689 train samples and 1 of 100
test samples.

### Probe

`probe.py` exposes a `HallucinationProbe(nn.Module)` and inside it I
keep five `sklearn.LogisticRegression(C=0.01)` instances trained on
bootstrap samples of the training data. Predictions are averaged. The
`nn.Module.forward` method has to exist by contract; I implement it by
averaging the five weight vectors and putting them in a single
`nn.Linear` so logits are roughly consistent with `predict_proba`. The
evaluator from the repo only calls `predict` and `predict_proba`, so
forward is just for shape correctness.

A few choices that mattered:

* `class_weight=None`, not `'balanced'`. The training prior is 70/30
  hallucinated to truthful, and the test set probably has a similar
  ratio. Balancing flattens the calibration and costs about 1 point of
  accuracy.
* Strong L2 (C=0.01). On a 689-by-1792 feature matrix, large C
  overfits. I checked C in {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0} and
  0.01 to 0.05 was the best range.
* Threshold tuning for accuracy, not for F1. F1 tuning ends up
  predicting more positives and accuracy drops by about 1 point.
* Threshold is selected from 5-fold out-of-fold probabilities inside
  `fit`. Every training sample contributes once to the threshold
  estimate. This is needed because the final probe in `solution.py` is
  trained without a separate `fit_hyperparameters` call and otherwise
  would have to use 0.5.

### Splitting

5-fold StratifiedKFold with `random_state=42`. All 689 prompts are
unique, no group-aware splits needed. Inside each outer fold I do an
80/20 stratified split of the train part with a different seed per
fold, so `evaluate.py` can use the held-out 20% for the per-fold
threshold tune. The result is 5 numbers per metric that the evaluator
averages.

## Reproducibility

Tested on Windows 11, Python 3.12, RTX 3070 8 GB with CUDA 12.4. Should
also run on Colab T4 or on CPU, just slower.

```bash
git clone https://github.com/Brekinilya/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Optional GPU wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch

pip install -r requirements.txt

python solution.py
```

`solution.py` will:

1. Download Qwen2.5-0.5B on first run (about 1 GB).
2. Forward all samples through the model and aggregate features.
3. Run 5-fold CV with `evaluate.py` and write `results.json`.
4. Train the final probe on all training data and write
   `predictions.csv` with predictions for `data/test.csv`.

On my machine the full run takes about 2 minutes.

Note: if the HuggingFace download stalls on the new Xet backend (I hit
this on Windows), set `HF_HUB_DISABLE_XET=1` before running. Then it
falls back to the legacy path and downloads in seconds.

The pipeline is deterministic on the same hardware. I checked that two
consecutive runs produce identical `predictions.csv`. Sources of
randomness are pinned: `splitting.split_data` uses `random_state=42`
with a per-fold offset, `HallucinationProbe.fit` uses fixed bootstrap
seeds `(0, 1, 7, 42, 123)`, and `aggregation.py` counts samples
deterministically from the CSV order.

## Experiments and ablations

The folder `experiments/` has 15 scripts. Each one operates on a
locally cached `.npz` archive of hidden states so the LLM does not have
to run more than a few times. The four caches are listed at the top of
each cache-building script (`extract_cache.py`, `_v2`, `_v3`, `_v4`).

| Script | What it does |
|---|---|
| `sweep_layer_pool.py` | Layer and pool sweep on full-sequence features. Found layer 14 last-token as the best v1 config. |
| `compare_probes.py` | Side by side comparison: baseline, multi-layer concat, Mass-Mean, MLP, geometric-only. |
| `tat_probe.py` | Truth as a Trajectory: LSTM over inter-layer displacement vectors. |
| `stability_and_threshold.py` | Layer 14 stays the best across 7 seeds. Sweeps `class_weight` and threshold-tuning metric. |
| `final_search.py` | Hyperparameter search around best v1 config. |
| `ensembles_and_gbm.py` | Multi-seed bootstrap, multi-layer averaging, LightGBM, hybrid LR+LGB. |
| `sweep_response_pool.py` | Layer and pool sweep on response-only features. Found L13 max-pool as new best. |
| `sweep_heuristic.py` | Heuristic last-K-percent-of-tokens vs exact response boundary. Heuristic loses 2 to 7 AUROC, so I built the counter-based tokenizer. |
| `response_ensemble.py` | Multi-seed bootstrap, LR+LGB hybrid, concat probes on response-only features. Bootstrap adds ~0.5 acc. |
| `multi_layer_response.py` | Independent LogReg per layer averaged across L12, L13, L17. Slightly worse accuracy than single L13 + bootstrap. |
| `extract_cache_v4.py` + `sweep_topk.py` | Top-K max-pool (K in 1,3,5,10,20). Top-10 wins on seed 42 but not in mean over 7 seeds. |
| `triple_pool_and_calibration.py` | Triple concat max+mean+first and isotonic calibration. Triple gives +0.66 AUROC but -1.6 accuracy; isotonic was a no-op. |
| `threshold_calibration.py` | Prior-aligned threshold strategies. Within ±0.1 of the accuracy-tuned threshold. |
| `c_search_and_l12l13.py` | Adopted: L12+L13 concat. Tested adaptive C-search too; within noise, kept fixed C. |

### What I tried and discarded

| Idea | Result | Why I dropped it |
|---|---|---|
| Mass-Mean Probe (Marks and Tegmark 2023) | AUROC ~58% | The "one truth direction" assumption is too restrictive for varied QA errors. |
| Truth as a Trajectory LSTM (Damirchi et al. 2026) | acc 66.92%, AUROC 71.25% | LSTM overfits on 689 samples, fold variance was huge. |
| Multi-layer concat of 5+ layers | acc 69.67% | Too many features for the sample size. |
| Multi-layer averaging of probabilities | -0.5 acc | The averaging dilutes confident predictions. |
| PCA into LogReg | AUROC 64-67% | The discriminative direction is in low-variance components. |
| MLP probe | acc 71-73% | Same accuracy as LogReg but with more hyperparameters and worse cross-seed stability. |
| LightGBM | acc 71-73% | Trees do not find useful nonlinearities here. |
| LR + LightGBM hybrid | acc 73.7% | Worse than LogReg alone. |
| Geometric features by themselves | acc 61.68% | Mostly captures the answer-length signal. |
| Geometric features on top of L14 | no gain | Already captured by the main feature. |
| Heuristic last-K-percent pool | up to 73.4% acc | Loses 2-7 AUROC vs exact response boundary. |
| Top-K max pool (K=3,5,10,20) | mean acc ~ same as K=1 | Wins on seed 42 only. AUROC slightly better but not enough. |
| Triple pool concat (max+mean+first) | acc 73.88%, AUROC 79.12% | +AUROC, -1.6 acc. I optimize accuracy. |
| Isotonic calibration of OOF probabilities | acc 75.47%, AUROC 77.73% | LogReg ranking is already good. |
| Prior-aligned threshold (acc + offset, quantile match) | within ±0.1 acc | Noise on stratified CV. |
| Adaptive C-search inside fit | +0.02 acc over fixed C=0.01 | Not worth the extra code. |
| `class_weight='balanced'` | -1 acc on every layer | Distorts calibration. |
| Threshold tuned for F1 | -1 acc | F1 prefers high recall on the positive class. |

### What actually helped

In order of accuracy contribution:

| Change | Approximate effect on 5-fold CV accuracy |
|---|---|
| Default skeleton -> layer 14 last token | +5.2 pp (62.3 -> 67-68) |
| Strong L2 (C=0.01 vs default C=1) | +1.6 pp |
| `class_weight=None` (was 'balanced') | +1.0 pp |
| Threshold tuned for accuracy (not F1) | +1.0 pp |
| Out-of-fold threshold tuning | better-behaved test predictions |
| Response-only max pool at layer 13 | +3.3 pp (71.99 -> 75.32) and +8 AUROC |
| 5-seed bootstrap | +0.5 pp |
| Concat L12 and L13 max-pools | +0.18 AUROC, halved per-fold std, predicted positive rate 88 -> 76 |
| 5-fold StratifiedKFold instead of one split | tighter mean, no accuracy change |

## Notes on the dataset

* 689 train samples and 100 test samples. All prompts are unique, so I
  do not need GroupKFold.
* Train class balance: 70.10% hallucinated, 29.90% truthful. So the
  majority baseline accuracy is 70.10%. Beating it by 5 points with a
  probe is the actual signal.
* Hallucinated answers are longer on average (mean 797 chars vs 421).
  This is a shallow but real signal. The response-only max-pool feature
  benefits indirectly because longer answers mean a bigger pool of
  activations to max over.
* The string "Unable to answer based on given context" appears in both
  classes (20 truthful, 31 hallucinated). So whether the model refused
  is not a clean label predictor by itself.
* Right-truncation at 512 tokens kills the entire response in 7 of 689
  training samples and 1 of 100 test samples. The aggregator falls back
  to the last real token in those cases.
