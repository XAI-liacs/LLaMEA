# LLaMEA RLM surrogate: data, model, and evaluation report

> **This report was generated from `tests/fixtures/rlm/`, a synthetic
> BLADE-schema fixture, not real LLaMEA/BLADE logs** -- the real data
> directory does not exist yet (see task Step 0). Its purpose is to prove
> the pipeline (`data_pipeline.py -> train.py -> evaluate.py -> report.py`)
> runs correctly end to end, and to document what the report looks like. It
> also used the offline `char`-encoder fallback config
> (`configs/tiny_local_test.yaml`), not the real frozen T5Gemma encoder the
> paper uses -- see the "Model setup notes" section below. **None of
> the numbers in this report should be treated as a real evaluation of the
> surrogate's usefulness.** Rerun the same four commands (below) against the
> real log directory once it's available.
>
> ```bash
> uv run python -m llamea.rlm_surrogate.data_pipeline --data-dir <logs> --output-dir data/
> uv run python -m llamea.rlm_surrogate.train --config llamea/rlm_surrogate/configs/default.yaml \
>     --train data/train.jsonl --val data/val.jsonl --output-dir checkpoints/run1
> uv run python -m llamea.rlm_surrogate.evaluate --checkpoint-dir checkpoints/run1 \
>     --train data/train.jsonl --test data/test.jsonl --output-dir results/eval_run1
> uv run python -m llamea.rlm_surrogate.report --stats data/stats.json \
>     --eval-results results/eval_run1/eval_results.json --config checkpoints/run1/config.yaml \
>     --output-dir results
> ```

## 1. Data

Ingested **98** records from **3** file(s)/run(s). Dropped **6** (6.1%) for a non-empty `error` field. **92** usable `(x, y)` examples remain (target = `fitness`, description included = True, configspace included = True).

### Per-file summary

| File | n | Errored (frac) | fitness min/max/mean | fitness skew | gen-vs-fitness Spearman | warnings |
|---|---|---|---|---|---|---|
| run_alpha | 48 | 4 (8.3%) | 0.706/1.000/0.938 | -1.669 | 0.320 | - |
| run_beta | 30 | 0 (0.0%) | 0.775/1.000/0.956 | -1.553 | 0.285 | - |
| run_gamma | 20 | 2 (10.0%) | 0.745/1.000/0.948 | -1.584 | 0.106 | - |

> **Warning:** Only 92 usable examples across 3 file(s). The RLM paper's strongest results come from tens-of-thousands to millions of examples; at this scale, treat correlation numbers as a few-shot/fine-tuning signal, not a trained-model guarantee, and prefer pooling multiple runs/files before trusting them.

## 2. Train/val/test split

Strategy: **whole_run_holdout_for_test**. This is deliberately *not* a random i.i.d. split: whole runs are held out for the test set whenever enough runs are available, and otherwise (or for the validation set) later generations within a run are held out rather than randomly interspersed, so near-duplicate mutated siblings of a training example can't leak into val/test.

- train: 38, val: 10, test: 44
- runs held out entirely for test: run_alpha
- per-run generation cutoff used for val: `{'run_beta': 4, 'run_gamma': 3}`

## 3. Model configuration

```yaml
encoder_type: char
t5gemma_model_name: google/t5gemma-s-s-prefixlm
freeze_encoder: true
d_model: 32
num_decoder_layers: 1
dropout: 0.0
max_input_len: 512
max_num_objs: 1
z_loss_coef: null
optimizer: adamw
lr: 0.003
weight_decay: 0.0
warmup_steps: 5
total_steps: null
max_epochs: 3
batch_size: 8
batch_size_per_device: null
patience: null
use_lora: false
lora_r: 8
lora_alpha: 16
lora_dropout: 0.0
seed: 0
num_samples_point_pred: 8
pretrained_checkpoint: null
```

## 4. Evaluation

Evaluated on **44** held-out examples (feature baseline fit on **38** train examples).

### 4.1 Overall ranking quality

| Arm | n | Spearman-rho | Kendall-tau |
|---|---|---|---|
| rlm | 44 | 0.346 | 0.239 |
| feature_baseline | 44 | 0.605 | 0.471 |
| random | 44 | 0.274 | 0.196 |

![Predicted vs true](predicted_vs_true.png)

### 4.2 Within-run ranking and selection accuracy (RLM)

This is the number that most directly answers whether pre-screening helps: for each run with enough held-out candidates, the within-run Spearman-rho and whether the model's top pick is the actual best-performing candidate in that run.

| Run | n | Spearman-rho | Top-1 hit | Top-25% overlap (RLM / random-expected) |
|---|---|---|---|---|
| run_alpha | 44 | 0.346 | no | 0.273 / 0.250 |


![Within-run Spearman](within_run_spearman.png)

### 4.3 Budget-reduction simulation

Given a generation of N candidates, evaluating only the top-k predicted by the RLM vs. k random candidates vs. all N:

| k/N | mean k | batches | RLM best-found frac. | Random best-found frac. | Compute savings |
|---|---|---|---|---|---|
| 0.10 | 1.0 | 6 | 0.974 | 0.942 | 90% |
| 0.25 | 2.0 | 6 | 0.983 | 0.974 | 75% |
| 0.50 | 3.7 | 6 | 0.984 | 0.992 | 50% |
| 0.75 | 5.3 | 6 | 0.997 | 0.997 | 25% |
| 1.00 | 7.3 | 6 | 1.000 | 1.000 | 0% |


![Budget reduction](budget_reduction.png)

## 5. Recommendation

- **Sample size caveat:** only 92 usable examples were used here. The RLM paper's strongest results come from tens-of-thousands to millions of examples; with a dataset this small, treat the correlation numbers above as a few-shot/fine-tuning signal, not a validated model. **Do not deploy this as a pre-screening gate until it has been re-evaluated on a pooled dataset of at least several hundred to low thousands of examples across multiple runs.**

- Only 1 run(s) had enough held-out candidates for within-run evaluation -- the budget-reduction numbers above should be treated as illustrative of the *method*, not as a reliable estimate of savings on a new run, until more runs are pooled.

- Overall Spearman-rho (0.35) is too low to recommend pre-screening at this dataset size/model configuration. Retrain with more pooled data and/or the real frozen T5Gemma encoder (this run may have used the lightweight from-scratch/offline fallback encoder) before reconsidering.

- Per-run fine-tuning (Step 2's `finetune_from_pool.yaml`) is worth testing once a pool of at least a few hundred historical examples across multiple runs exists; its value can't be assessed from a single run's data alone.

- No field in the current BLADE schema identifies the benchmark function/problem for each candidate, so ranking quality could only be broken out by run, not by problem. If per-function breakdown is needed, add a `problem_id`-style field to the logger.

## Model setup notes

- **Real frozen pretrained encoder (`configs/default.yaml`, `encoder_type: t5gemma`):**
  requires the `rlm-surrogate` dependency group (`uv sync --group rlm-surrogate`,
  which installs `torch`, `transformers`, `peft`, and `regress-lm` from
  `git+https://github.com/google-deepmind/regress-lm.git` -- **the `regress-lm`
  package on PyPI, version `0.0.1`, is an unrelated empty placeholder and must
  not be used**; `pyproject.toml`'s `[tool.uv.sources]` already points at the
  real GitHub source) and a HuggingFace account with the Gemma license
  accepted (`huggingface-cli login`). `google/t5gemma-s-s-prefixlm` is gated;
  attempting to load it without authentication fails with an `OSError` (as
  observed while building this pipeline in a sandboxed environment with no HF
  credentials).
- **Offline fallback (`configs/tiny_local_test.yaml`, `encoder_type: char`):**
  a small from-scratch transformer with a local character vocabulary, no
  network access required. Used to produce this report and by the automated
  test suite (`tests/test_rlm_pipeline_e2e.py`). It is **not** the paper's
  frozen-pretrained-encoder architecture and should not be used to judge
  real-world surrogate quality.
- **`encoder_type: vanilla`:** regress-lm's small from-scratch transformer
  with the public T5 SentencePiece vocab (downloaded from GCS on first use,
  network required but no HF gating). Also not a pretrained encoder --
  useful as a middle ground if T5Gemma access isn't available but network
  access is.
- **Fine-tuning from a pool (`configs/finetune_from_pool.yaml`):** point
  `pretrained_checkpoint` at a checkpoint produced by a prior `train.py` run
  over pooled historical logs, then train on the new run's split. Not
  exercised in this report since only one synthetic dataset was available.
