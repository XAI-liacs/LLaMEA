# Running the RLM surrogate on a GPU cluster

This is a runbook for taking the pipeline in this directory from a clean
GPU machine to a finished `results/report.md`, with the real frozen
T5Gemma encoder the paper uses (not the CPU-only fallback described below).

Everything here was built and exercised end-to-end against the real
`BLADE-results` data (BBOB + MA-BBOB, ~15.8k records) in a **CPU-only**,
network-restricted sandbox, using the offline `configs/blade_results_cpu.yaml`
fallback (`encoder_type: vanilla`, `d_model: 128`, `max_input_len: 512`) --
see `results/report.md` for that dry run's findings, including two real
data quirks worth knowing about before you start:

- **Median tokenized input length in this dataset is ~3,200 SentencePiece
  tokens** (code bodies are long). `max_input_len: 512` in the CPU config
  truncates most examples heavily -- on a GPU, raise this substantially
  (see Step 2 below).
- One log folder (`MA_BBOB`) never populates the `error` field on failure --
  it only shows up as `fitness: -inf` with exception text in `feedback`.
  `BladeRecord.is_invalid` (`schema.py`) already catches this; you don't
  need to do anything, just know that "error rate" in `stats.json` reflects
  this broader check, not a literal `error != ""` count.

## 1. Prerequisites

- **A CUDA GPU.** The paper's smallest T5Gemma variant
  (`google/t5gemma-s-s-prefixlm`) is the reasonable starting point; a
  bigger GPU buys you a larger `max_input_len`/`batch_size`, not a
  requirement to change model size.
- **HuggingFace access to the gated T5Gemma model.** This is the one
  prerequisite the sandbox run could not satisfy (confirmed live: loading
  `google/t5gemma-s-s-prefixlm` without credentials fails with an
  `OSError`). Before training:
  1. Accept the license on the model page:
     https://huggingface.co/google/t5gemma-s-s-prefixlm
  2. Authenticate on the GPU machine: `huggingface-cli login`, or set
     `HF_TOKEN` in the environment.
- **Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).**

## 2. Install

```bash
git clone https://github.com/XAI-liacs/LLaMEA.git
cd LLaMEA
git checkout claude/llamea-rlm-surrogate-20but4   # or main, once PR #139 merges
uv sync --group rlm-surrogate
# add `--dev` too if you also want to run the test suite (pytest/black/isort)
```

`pyproject.toml` already pins `regress-lm` to
`git+https://github.com/google-deepmind/regress-lm.git` via
`[tool.uv.sources]` -- **do not** change this to a plain `regress-lm`
PyPI dependency. The PyPI package of that name (version `0.0.1`) is an
unrelated empty placeholder distribution with no actual code in it; this
was discovered the hard way earlier in this project and the git-source pin
is the fix.

## 3. Get the data onto the cluster

Copy your `BLADE-results` tree (or wherever your LLaMEA/BLADE logs live)
onto the GPU machine, e.g.:

```bash
rsync -avz /local/path/to/BLADE-results/ gpu-box:/data/BLADE-results/
```

The pipeline supports two layouts (`data_pipeline.py --layout`):

- `flat`: `--data-dir` directly contains `.jsonl` run files (one file per
  run). This is what the unit tests/fixtures use.
- `per_problem_subdir`: `--data-dir` contains
  `<experiment_folder>/run-*/log.jsonl` -- **this is what the real
  `BLADE-results` export uses**, and is the one you want. `problem_id` is
  derived per experiment folder (`BBOB` vs `MA-BBOB`, via
  `classify_problem` in `data_pipeline.py`); `conversationlog.jsonl`,
  `experimentlog.jsonl`, and `progress.json` are ignored.

## 4. Step 1 -- data pipeline

```bash
uv run python -m llamea.rlm_surrogate.data_pipeline \
    --data-dir /data/BLADE-results \
    --output-dir data/ \
    --layout per_problem_subdir
```

This writes `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, and
`data/stats.json`. **Read `stats.json` before moving on** -- it has
per-run and per-problem record counts, error fractions, and fitness
range/skew; a per-file `warnings` list flags anything that looks off (high
error rate, an out-of-[0,1]-range fitness convention, etc.).

Useful flags:
- `--exclude-dir-substring SUBSTRING` (repeatable): skip experiment folders
  whose name contains `SUBSTRING`, case-insensitive. Defaults to excluding
  anything with "debug" or "wrong" in the name. Pass
  `--exclude-dir-substring ''` once to include everything instead.
- `--target {fitness,aucs,aucs_per_instance}`: train against the scalar
  `fitness` (default), the whole `metadata.aucs` sequence as one
  multi-objective example, or explode each candidate into one example per
  `aucs[i]` with a real problem-landscape fingerprint in `x` -- see
  "Problem features (optional)" below.
- `--no-description` / `--no-configspace`: code-only ablation (drop the
  prepended context from `x`).
- `--test-run-fraction`, `--min-runs-for-file-holdout`, `--val-fraction`,
  `--test-fraction`, `--seed`: control the lineage-/generation-aware split
  (see `SplitConfig` in `data_pipeline.py`). Defaults are reasonable for
  datasets with dozens+ runs per problem, which this dataset has.

## 5. Problem features (optional): `--target aucs_per_instance`

By default `x` = code (+ description/configspace) only -- nothing tells
the model *which* landscape a candidate was scored on beyond the coarse
`BBOB`/`MA-BBOB` `problem_id`. `--target aucs_per_instance` fixes that:
each candidate's `metadata.aucs` (its per-instance AOCC breakdown -- the
scalar `fitness` you'd otherwise train on is just `mean(aucs)`) is
exploded into one training example per instance, each with a Latin
Hypercube fingerprint of the *actual* reconstructed BBOB/MA-BBOB function
appended to `x`. This is what turns "predict this candidate's aggregate
score" into "predict this candidate's score on *this* landscape" -- the
only way to get real cross-problem generalization (e.g. zero-shot to a
BBOB function never seen in training) rather than memorizing per-problem
score ranges.

```bash
uv run python -m llamea.rlm_surrogate.data_pipeline \
    --data-dir /data/BLADE-results \
    --output-dir data_instances/ \
    --layout per_problem_subdir \
    --target aucs_per_instance \
    --lhs-points 20
```

Requires `ioh` (already in the `rlm-surrogate` group as of this README).

**How instance identity is resolved (no guessing).** Reconstructing the
exact landscape behind `aucs[i]` needs to know which `(dim, function,
instance)` each entry corresponds to. `problem_instances.py` reads this
from real metadata, in order:

1. **`metadata.performance_data`** (BBOB records only): present on every
   non-errored BBOB record in the datasets checked. Each entry
   self-describes its instance -- `{"fid": 1, "iid": 1, "dim": 10, "auc":
   0.85}` -- aligned 1:1 with `aucs`. No external file needed.
2. **The sibling `experimentlog.jsonl`** (one level up from each
   `<experiment>/run-*/log.jsonl`, keyed by `log_dir` == the run folder
   name): used for every MA_BBOB record (which never carries
   `performance_data`) and as the BBOB fallback. Its `problem` block gives
   `problem.dims` (a single-element list) and `problem.training_instances`
   -- for BBOB a literal list of `[fid, iid]` pairs, for MA_BBOB a string
   like `"range(0, 10)"` -- aligned 1:1 with that run's `aucs`.

A record is skipped (never guessed) and counted in `stats.json`'s
`instance_explosion` block when neither source resolves
(`n_no_instance_mapping`) or the resolved instance list's length doesn't
match `len(aucs)` (`n_length_mismatch`); `n_no_aucs` counts records with no
`metadata.aucs` at all, and `n_instance_errors` counts per-instance
reconstruction failures (e.g. a bad MA-BBOB table row). Check these counts
after a run to see how much of your data this feature could actually use.

`evaluate.py`/`report.py` handle exploded data automatically: candidate-level
metrics (Spearman, within-run ranking, budget-reduction) still work
unchanged by averaging each candidate's per-instance predictions back
together first (`aggregate_instance_predictions`); an added
"Instance-level generalization" section in the report shows the raw,
un-aggregated per-instance Spearman -- the direct signal of whether the
model distinguishes landscapes at all.

## 6. Step 2 -- training

Two starting points, both in `configs/`:

### 6a. Train from scratch on the full pooled data

```bash
uv run python -m llamea.rlm_surrogate.train \
    --config llamea/rlm_surrogate/configs/default.yaml \
    --train data/train.jsonl --val data/val.jsonl \
    --output-dir checkpoints/base
```

`configs/default.yaml` uses `encoder_type: t5gemma`, `freeze_encoder: true`
-- the paper's actual setup. **Before running it on real data, edit
`max_input_len`**: the shipped value (2048) is closer to right than the
CPU config's 512, but given the ~3.2k-token median in this dataset,
consider pushing it to 4096+ if your GPU memory allows (self-attention
cost grows roughly quadratically with this, so check memory/throughput
before committing to a big run). Other knobs worth adjusting for GPU
throughput:
- `batch_size` (was `null` = full-batch in the CPU config; set an actual
  number) and `batch_size_per_device` (for gradient accumulation if the
  full batch doesn't fit).
- `d_model`/`num_decoder_layers` only affect the from-scratch encoder
  paths (`vanilla`/`char`) -- not `t5gemma`, whose encoder width is fixed
  by the pretrained checkpoint.
- `max_epochs`/`patience`/`warmup_steps` -- the CPU config used
  `max_epochs: 6` purely to fit in a CPU time budget; the paper's Appendix
  C defaults (in `configs/default.yaml` already) are a better starting
  point with real compute.

### 6b. Pretrain on a pool, fine-tune per problem

Once you have a checkpoint from 5a trained on pooled historical data,
fine-tune it on a specific problem's (typically early-generation)
examples:

```bash
uv run python -m llamea.rlm_surrogate.train \
    --config llamea/rlm_surrogate/configs/finetune_from_pool.yaml \
    --pretrained-checkpoint checkpoints/base/model.pt \
    --train data/train.jsonl --val data/val.jsonl \
    --output-dir checkpoints/finetuned
```

`configs/finetune_from_pool.yaml` uses a smaller LR/fewer epochs (adapting,
not learning from scratch) and already has `pretrained_checkpoint` wired
up in the YAML; `--pretrained-checkpoint` on the CLI overrides it if you'd
rather not edit the file. This mirrors the paper's NAS fine-tuning result
(Table 13) and is the config to reach for once a pool of historical
examples across multiple runs/problems exists -- which, at ~15.8k real
records across BBOB and MA-BBOB, this dataset now does.

Each run writes `model.pt`, `config.yaml` (the resolved config, needed by
`evaluate.py` to rebuild the same architecture), and `train_manifest.json`
to `--output-dir`.

**Don't fine-tune from the CPU sandbox's checkpoint.** It used the offline
`vanilla` encoder (not T5Gemma) at reduced width specifically to fit a CPU
time budget -- start fresh with 5a on the GPU box instead.

## 7. Step 3 -- evaluate

```bash
uv run python -m llamea.rlm_surrogate.evaluate \
    --checkpoint-dir checkpoints/base \
    --train data/train.jsonl \
    --test data/test.jsonl \
    --output-dir results/eval_base
```

Needs `--train` too (used to fit the hand-featured baseline regressor, for
comparison). Writes `results/eval_base/eval_results.json`: overall
Spearman/Kendall, a per-run and per-problem breakdown, within-run
top-k selection accuracy vs. random, and the budget-reduction (evaluate
top-k of N vs. random-k vs. all-N) simulation. Add `--no-baselines` to
skip the feature/random baselines if you only want the RLM numbers.

## 8. Step 4 -- report

```bash
uv run python -m llamea.rlm_surrogate.report \
    --stats data/stats.json \
    --eval-results results/eval_base/eval_results.json \
    --config checkpoints/base/config.yaml \
    --output-dir results
```

Writes `results/report.md` plus `results/figures/*.png`
(predicted-vs-true scatter, within-run Spearman-by-run bar chart,
budget-reduction curve). This is a straight regeneration of the same
report structure as the CPU dry run, just with real numbers this time --
diff it against the current `results/report.md` in this repo to see
exactly what changes.

## 9. Keep data and checkpoints out of git

`data/train.jsonl` alone was ~90MB on the real dataset at CPU-fallback
settings, and checkpoints/full `eval_results.json` predictions payloads
add more. None of that belongs in this repo. If you're working in a clone
of it, add to `.gitignore` (or just keep these directories outside the
clone entirely, as this session did):

```gitignore
/data/
/checkpoints/
results/eval_*/
```

`results/report.md` and `results/figures/*.png` are small and fine to
commit -- that's the actual deliverable.
