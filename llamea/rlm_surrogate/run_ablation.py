"""Problem-feature ablation: trains + evaluates the RLM surrogate under a
few different "# Problem" feature representations (see
``problem_instances.compute_problem_feature_text``'s ``mode`` menu) on a
*subsampled* dataset, so you can rank variants before committing the full
compute budget to whichever wins.

Evaluation uses a leave-function-out test split
(``data_pipeline.leave_function_out_split``) rather than the default
lineage/generation split: the whole point of adding problem features is
generalization to a landscape the model never trained on, and a within-run
split can't tell "learned to use problem features" apart from "memorized
this function's score range."

Default variants (3 runs, matching the ones actually implemented so far):
  - ``lhs``            : current shipped default (raw LHS samples only)
  - ``meta+lhs``        : Tier A static properties + raw LHS
  - ``meta+lhs_stats``  : Tier A static properties + Tier B computed stats

Runs the full ``variant x seed x holdout_fids`` matrix (default: 3 variants
x 5 seeds x 3 holdout sets = 45 runs) so a variant's apparent edge can be
checked against seed-to-seed noise and against more than one held-out
region of the BBOB taxonomy, rather than trusting one run on one holdout
pair -- a real run of the earlier single-seed/single-holdout version showed
two variants within 0.003 Spearman of each other, well inside plausible
single-seed noise. ``ablation_summary.json``'s ``by_variant`` block reports
each variant's mean/std across every (seed, holdout) run so you can see
whether a gap survives that variance instead of eyeballing individual
numbers.

**Wall-clock warning**: the default matrix is 45 runs. A single run at the
(also-raised) default training budget will take considerably longer than
the earlier short-budget ablation (which itself took ~12h/run at
max_epochs=10/max_steps_per_epoch=200 on real hardware) -- run the full
default matrix serially only if you have days to spare. Shard the work
instead: launch several CLI invocations with disjoint ``--seeds``/
``--variants``/``--holdout-fids`` subsets (one per GPU/machine you have),
or explicitly pass smaller ``--seeds``/``--holdout-fids`` lists and fewer
``--max-epochs``/``--max-steps-per-epoch`` for a quicker pass.

**Restartable by default.** Before running a ``(variant, seed,
holdout_fids)`` combination, ``run_ablation`` checks whether that run's
directory already has a complete, valid ``ablation_result.json`` and, if
so, skips straight to reusing it instead of retraining. This makes it safe
to re-run the exact same command after an interruption (Ctrl-C, OOM,
preemption, a crashed process) -- already-finished runs are loaded from
disk instead of redone, and only what's missing or was left incomplete
actually runs. This does *not* resume a single run mid-training from a
checkpoint -- a run without a saved ``ablation_result.json`` (never
started, or interrupted partway through) is simply redone from scratch.
Pass ``--force-rerun`` to ignore any cached results and redo everything
(e.g. after a code change that would invalidate old numbers).

Requires the ``ioh`` extra, real ``BLADE-results`` data in the
``per_problem_subdir`` layout, and a GPU for the T5Gemma config -- this is a
driver, not something exercised in the (CPU-only, synthetic-fixture) test
suite.

CLI:
    uv run python -m llamea.rlm_surrogate.run_ablation \\
        --data-dir /data/BLADE-results --output-dir results/ablation \\
        --max-records 10000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from .config import RLMSurrogateConfig
from .data_pipeline import SplitConfig, run_pipeline_multi_problem

DEFAULT_VARIANTS = ["lhs", "meta+lhs", "meta+lhs_stats"]

# 5 seeds gives enough spread to tell a real effect from single-seed noise
# without exploding runtime further than the holdout/variant axes already do.
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Three pairs spanning different COCO/BBOB groups (see bbob_properties.py's
# GROUP_NAMES), so a variant's apparent edge isn't just an artifact of one
# specific pair of held-out landscapes:
#   [21, 22] : group 5, multi-modal weak structure (Gallagher's Peaks) --
#              the original single-holdout ablation's pair.
#   [3, 8]   : group 1 separable multi-modal (Rastrigin) + group 2
#              moderate-conditioning (Rosenbrock) -- a differently-shaped
#              pair than the other two sets.
#   [13, 19] : group 3 high-conditioning unimodal (Sharp Ridge) + group 4
#              moderate multi-modal (Griewank-Rosenbrock).
DEFAULT_HOLDOUT_SETS: list[list[int]] = [[21, 22], [3, 8], [13, 19]]


def _run_dir_for(
    output_dir: Path, label: str, seed: int, holdout_fids: list[int]
) -> Path:
    """Single source of truth for a run's directory name -- shared by the
    function that writes there (``run_one_variant``) and the resume check
    in ``run_ablation`` (which needs the same path *before* deciding
    whether to call ``run_one_variant`` at all). ``label`` is the
    variant/config identifier (e.g. ``"meta_lhs"`` here, ``"lhs50"`` in
    ``run_ablation_lhs_points.py``, which reuses this helper)."""
    holdout_tag = "-".join(str(f) for f in holdout_fids)
    return output_dir / f"{label}__seed{seed}__holdout{holdout_tag}"


def _write_json_atomic(path: Path, data: Any) -> None:
    """Writes JSON to a temp file then renames it into place, so a process
    killed mid-write (Ctrl-C, OOM-killer, preemption) never leaves a
    truncated/corrupt file at ``path`` -- which the resume check below
    would otherwise either crash on or (worse) silently treat as a
    complete result and skip re-running."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp_path, path)


# Keys a genuinely completed run_one_variant/run_one_lhs_points_variant
# result must have -- guards against treating a result file from an old,
# differently-shaped schema (or a hand-edited/truncated one) as valid.
_REQUIRED_RESULT_KEYS = {"spearman_rho", "kendall_tau", "n_test", "wall_clock_seconds"}


def _load_cached_result(run_dir: Path) -> dict[str, Any] | None:
    """Returns the previously-saved result for ``run_dir`` if it looks like
    a complete, valid run, else ``None`` -- covering "never started",
    "crashed/killed partway through" (no result file saved yet), and "the
    result file is present but corrupt or from an incompatible old
    schema" the same way: treat it as not done, and let the caller redo it
    from scratch. This is what makes both ablation scripts restartable --
    note it never resumes a single run mid-training from a checkpoint, it
    only skips runs that already finished."""
    result_path = run_dir / "ablation_result.json"
    if not result_path.exists():
        return None
    try:
        with open(result_path) as fh:
            result = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(result, dict) or not _REQUIRED_RESULT_KEYS.issubset(result):
        return None
    return result


def _release_gpu_memory() -> None:
    """Runs between train/eval stages and between variants.

    Neither ``train.py`` nor ``evaluate.py`` explicitly frees their
    model/optimizer state -- ``run_one_variant`` builds a fresh RLM for
    training and another for evaluation, and ``run_ablation`` calls it
    repeatedly in the same process, so a previous stage's/variant's
    memory can still be resident (and fragmented, per a real confirmed
    OOM: "10.79 GiB memory in use" of which only 6.66 GiB was actually
    allocated to live tensors, the rest reserved-but-unused by PyTorch's
    caching allocator) when the next one starts to allocate. `gc.collect()`
    clears out-of-scope Python references so CUDA tensors are actually
    freed, then `torch.cuda.empty_cache()` returns the now-unused cached
    blocks to the driver so the next allocation isn't fighting
    fragmentation from a memory pool with capacity to spare overall but
    no single free block large enough.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _short_config(
    base_config_path: str | Path,
    *,
    max_epochs: int,
    max_steps_per_epoch: int,
    patience: int,
) -> RLMSurrogateConfig:
    """Loads ``base_config_path`` (normally ``configs/default.yaml``) and
    overrides just the training-budget knobs down to a size meant for
    ranking variants quickly, not for convergence -- once a winner is
    picked, rerun it with the base config's real budget."""
    config = RLMSurrogateConfig.from_yaml(base_config_path)
    return config.with_overrides(
        max_epochs=max_epochs,
        max_steps_per_epoch=max_steps_per_epoch,
        patience=patience,
    )


def run_one_variant(
    *,
    variant: str,
    seed: int,
    data_dir: str | Path,
    output_dir: Path,
    holdout_fids: list[int],
    max_records: int,
    n_lhs_points: int,
    base_config_path: str | Path,
    max_epochs: int,
    max_steps_per_epoch: int,
    patience: int,
    include_baselines: bool,
    predict_batch_size: int,
) -> dict[str, Any]:
    from . import evaluate as evaluate_module
    from . import train as train_module

    run_dir = _run_dir_for(output_dir, variant.replace("+", "_"), seed, holdout_fids)
    data_out = run_dir / "data"
    checkpoint_dir = run_dir / "checkpoint"

    t0 = time.time()
    pipeline_summary = run_pipeline_multi_problem(
        data_dir,
        data_out,
        target="aucs_per_instance",
        feature_mode=variant,
        holdout_fids=holdout_fids,
        max_records=max_records,
        n_lhs_points=n_lhs_points,
        split_config=SplitConfig(seed=seed),
    )
    t_pipeline = time.time() - t0

    config = _short_config(
        base_config_path,
        max_epochs=max_epochs,
        max_steps_per_epoch=max_steps_per_epoch,
        patience=patience,
    )
    config = config.with_overrides(seed=seed)

    t0 = time.time()
    train_module.train(
        config,
        data_out / "train.jsonl",
        data_out / "val.jsonl",
        checkpoint_dir,
    )
    t_train = time.time() - t0

    # train_module.train() builds its own RLM (model + optimizer state) that
    # goes out of scope here; without an explicit collect+empty_cache, that
    # GPU memory can still be resident/fragmented when evaluate_module below
    # loads a second RLM for evaluation (confirmed real OOM: see
    # `_release_gpu_memory`'s docstring).
    _release_gpu_memory()

    t0 = time.time()
    eval_report = evaluate_module.run_full_evaluation(
        checkpoint_dir,
        data_out / "train.jsonl",
        data_out / "test.jsonl",
        include_baselines=include_baselines,
        seed=seed,
        predict_batch_size=predict_batch_size,
    )
    t_eval = time.time() - t0

    rlm_overall = eval_report["arms"]["rlm"]["overall"]
    result = {
        "variant": variant,
        "seed": seed,
        "holdout_fids": list(holdout_fids),
        "n_train": pipeline_summary["split"]["n_train"],
        "n_val": pipeline_summary["split"]["n_val"],
        "n_test": pipeline_summary["split"]["n_test"],
        "instance_explosion": pipeline_summary.get("instance_explosion", {}),
        "spearman_rho": rlm_overall["spearman_rho"],
        "kendall_tau": rlm_overall["kendall_tau"],
        "instance_level": eval_report.get("instance_level", {}).get("rlm"),
        "wall_clock_seconds": {
            "pipeline": t_pipeline,
            "train": t_train,
            "eval": t_eval,
        },
        "run_dir": str(run_dir),
    }
    _write_json_atomic(run_dir / "ablation_result.json", result)
    return result


def _summarize_by_variant(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregates spearman/kendall across every (seed, holdout_fids) run per
    variant -- mean and sample standard deviation, so a gap between variants
    can be checked against run-to-run spread instead of eyeballing single
    numbers (a real single-seed/single-holdout ablation showed two variants
    within 0.003 Spearman of each other -- well within plausible noise)."""
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = {}
    for variant, runs in by_variant.items():
        spearmans = [r["spearman_rho"] for r in runs]
        kendalls = [r["kendall_tau"] for r in runs]
        summary[variant] = {
            "n_runs": len(runs),
            "spearman_mean": statistics.mean(spearmans),
            "spearman_std": statistics.stdev(spearmans) if len(spearmans) > 1 else 0.0,
            "kendall_mean": statistics.mean(kendalls),
            "kendall_std": statistics.stdev(kendalls) if len(kendalls) > 1 else 0.0,
        }
    return summary


def run_ablation(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    variants: list[str] = DEFAULT_VARIANTS,
    seeds: list[int] = DEFAULT_SEEDS,
    holdout_sets: list[list[int]] = DEFAULT_HOLDOUT_SETS,
    max_records: int = 6000,
    n_lhs_points: int = 20,
    base_config_path: str | Path = Path(__file__).parent / "configs" / "default.yaml",
    max_epochs: int = 20,
    max_steps_per_epoch: int = 300,
    patience: int = 6,
    include_baselines: bool = False,
    predict_batch_size: int = 4,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Runs every ``(variant, seed, holdout_fids)`` combination and writes a
    summary table (including a ``by_variant`` mean/std rollup, see
    ``_summarize_by_variant``) to ``output_dir/ablation_summary.json``.
    Returns the list of per-run result dicts (also what gets written).

    ``resume`` (default ``True``): before running a combination, check
    whether its run directory already has a complete, valid
    ``ablation_result.json`` (see ``_load_cached_result``) and, if so,
    reuse it instead of retraining -- makes rerunning the same command
    after an interruption pick up only what's missing. Pass ``False`` to
    ignore any cached results and redo everything.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in variants:
        for seed in seeds:
            for holdout_fids in holdout_sets:
                holdout_fids = list(holdout_fids)
                run_dir = _run_dir_for(
                    output_dir, variant.replace("+", "_"), seed, holdout_fids
                )
                if resume:
                    cached = _load_cached_result(run_dir)
                    if cached is not None:
                        print(
                            f"=== variant={variant!r} seed={seed} "
                            f"holdout_fids={holdout_fids} -- SKIPPED (already "
                            f"completed, found {run_dir / 'ablation_result.json'}) ==="
                        )
                        results.append(cached)
                        continue
                print(
                    f"=== variant={variant!r} seed={seed} "
                    f"holdout_fids={holdout_fids} ==="
                )
                result = run_one_variant(
                    variant=variant,
                    seed=seed,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    holdout_fids=holdout_fids,
                    max_records=max_records,
                    n_lhs_points=n_lhs_points,
                    base_config_path=base_config_path,
                    max_epochs=max_epochs,
                    max_steps_per_epoch=max_steps_per_epoch,
                    patience=patience,
                    include_baselines=include_baselines,
                    predict_batch_size=predict_batch_size,
                )
                results.append(result)
                # Same rationale as the train/eval cleanup inside
                # run_one_variant: without this, one run's leftover GPU
                # memory can still be resident (and fragmented) when the
                # next (variant, seed, holdout_fids) starts training a
                # fresh RLM.
                _release_gpu_memory()
                print(
                    f"  -> spearman={result['spearman_rho']:.3f} "
                    f"kendall={result['kendall_tau']:.3f} "
                    f"n_test={result['n_test']} "
                    f"({sum(result['wall_clock_seconds'].values()):.0f}s)"
                )

    _write_json_atomic(
        output_dir / "ablation_summary.json",
        {
            "holdout_sets": [list(h) for h in holdout_sets],
            "max_records": max_records,
            "variants": variants,
            "seeds": list(seeds),
            "by_variant": _summarize_by_variant(results),
            "results": results,
        },
    )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        required=True,
        help="Root dir in the per_problem_subdir layout (same as "
        "data_pipeline.py --layout per_problem_subdir).",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=["lhs", "lhs_stats", "meta", "meta+lhs", "meta+lhs_stats"],
        help=f"Which feature modes to compare. Default: {DEFAULT_VARIANTS} "
        "(3 runs).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Repeat each (variant, holdout_fids) combination for each of "
        f"these seeds. Default: {DEFAULT_SEEDS} -- enough spread to tell a "
        "real effect from single-seed noise (a real single-seed ablation "
        "showed two variants within 0.003 Spearman of each other).",
    )
    p.add_argument(
        "--holdout-fids",
        type=int,
        nargs="+",
        action="append",
        dest="holdout_sets",
        default=None,
        help="BBOB function ids held out entirely for test (leave-function-"
        "out split) -- pass this flag multiple times to test more than one "
        f"holdout set, e.g. `--holdout-fids 21 22 --holdout-fids 3 8`. "
        f"Default (if omitted entirely): {DEFAULT_HOLDOUT_SETS} -- three "
        "pairs spanning different COCO/BBOB groups (see module docstring) "
        "so a variant's edge isn't just an artifact of one held-out region.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=6000,
        help="Subsample size for the fast ablation pass -- not the full "
        "dataset. Raising this raises the *exploded* eval/test set size "
        "roughly proportionally (each record becomes several instance "
        "rows), which raises prediction cost roughly proportionally too "
        "(see --predict-batch-size) -- keep this modest (low thousands) "
        "for a fast comparison; do a full run separately once you've "
        "picked a winning variant.",
    )
    p.add_argument("--lhs-points", type=int, default=20, dest="n_lhs_points")
    p.add_argument(
        "--base-config",
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
        dest="base_config_path",
        help="Model/optimizer config to start from; only max_epochs/"
        "max_steps_per_epoch/patience are overridden for the short "
        "ablation budget.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Default raised from the original fast-ranking pass (10) for "
        "a more realistic training effort -- still well short of the "
        "production config's 60, since this multiplies by seeds x "
        "holdout sets x variants. Lower it back down for a quick smoke "
        "test of the harness itself.",
    )
    p.add_argument("--max-steps-per-epoch", type=int, default=300)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also compute the feature/random baselines per run (slower, "
        "not needed just to rank variants against each other).",
    )
    p.add_argument(
        "--predict-batch-size",
        type=int,
        default=4,
        help="Chunk size for RLM sampling-based prediction during "
        "evaluation -- bounds memory/time independent of eval-set size "
        "(each chunk expands to predict_batch_size * "
        "num_samples_point_pred sequences). Default is deliberately "
        "conservative (confirmed on real hardware: 32 at max_input_len=4096 "
        "needed ~16GB just for one internal tensor). Raise it only after "
        "confirming GPU headroom; lower it further if you still OOM.",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore any existing ablation_result.json files and redo every "
        "(variant, seed, holdout_fids) combination from scratch, instead "
        "of the default resume behavior (skip combinations that already "
        "have a complete result). Use this after a code change that would "
        "invalidate previously cached numbers.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    holdout_sets = (
        args.holdout_sets if args.holdout_sets is not None else DEFAULT_HOLDOUT_SETS
    )
    results = run_ablation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        variants=args.variants,
        seeds=args.seeds,
        holdout_sets=holdout_sets,
        max_records=args.max_records,
        n_lhs_points=args.n_lhs_points,
        base_config_path=args.base_config_path,
        max_epochs=args.max_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        patience=args.patience,
        include_baselines=args.include_baselines,
        predict_batch_size=args.predict_batch_size,
        resume=not args.force_rerun,
    )
    print("\n=== Ablation summary (individual runs) ===")
    for r in sorted(results, key=lambda r: -r["spearman_rho"]):
        print(
            f"{r['variant']:<16} seed={r['seed']} holdout={r['holdout_fids']} "
            f"spearman={r['spearman_rho']:.3f} kendall={r['kendall_tau']:.3f} "
            f"n_test={r['n_test']}"
        )

    print("\n=== Ablation summary (by variant, mean +/- std) ===")
    by_variant = _summarize_by_variant(results)
    for variant, stats in sorted(
        by_variant.items(), key=lambda kv: -kv[1]["spearman_mean"]
    ):
        print(
            f"{variant:<16} n_runs={stats['n_runs']} "
            f"spearman={stats['spearman_mean']:.3f}+/-{stats['spearman_std']:.3f} "
            f"kendall={stats['kendall_mean']:.3f}+/-{stats['kendall_std']:.3f}"
        )


if __name__ == "__main__":
    main()
