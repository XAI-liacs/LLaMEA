"""Problem-feature ablation #2: varies the LHS-sample *density*
(``n_lhs_points``, see ``problem_instances._lhs_sample``) rather than the
feature-*mode* text representation ``run_ablation.py`` ablates. Answers a
different question: given a fixed way of describing the problem (default
``"lhs"``: raw ``(x)->f(x)`` pairs), does sampling more or fewer points per
problem instance change how well the RLM surrogate generalizes?

This is a deliberately separate, parallel script -- it does not modify
``run_ablation.py``, only imports a few small, already-tested utilities from
it (`_release_gpu_memory`, `_short_config`, `_summarize_by_variant`, and the
`DEFAULT_SEEDS`/`DEFAULT_HOLDOUT_SETS` constants) so the two ablations run
under identical seed and holdout conditions and are directly comparable,
without duplicating that logic.

Uses the same leave-function-out test split
(``data_pipeline.leave_function_out_split``) and the same
``variant x seed x holdout_fids`` matrix structure as ``run_ablation.py``,
with ``n_lhs_points`` playing the role of "variant": for the same reason
(single-run numbers can't be trusted -- a real feature-mode ablation showed
two variants within 0.003 Spearman of each other), rank LHS-density options
across several seeds and several held-out regions rather than one run each.

Default: 3 point-count variants (``5``/``20``/``50`` -- sparse, the current
shipped default, and denser) x 5 seeds x 3 holdout-fid sets = 45 runs, same
order of magnitude as ``run_ablation.py``'s default matrix.

**Wall-clock warning**: same caveat as ``run_ablation.py`` -- the default
45-run matrix at a realistic training budget takes a long time serially.
Shard it (disjoint ``--seeds``/``--lhs-points``/``--holdout-fids`` per
GPU/machine) or pass smaller lists / a smaller ``--max-epochs``/
``--max-steps-per-epoch`` for a quicker pass.

Requires the ``ioh`` extra, real ``BLADE-results`` data in the
``per_problem_subdir`` layout, and a GPU for the T5Gemma config -- this is a
driver, not something exercised in the (CPU-only, synthetic-fixture) test
suite.

CLI:
    uv run python -m llamea.rlm_surrogate.run_ablation_lhs_points \\
        --data-dir /data/BLADE-results --output-dir results/ablation_lhs_points \\
        --max-records 10000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .data_pipeline import SplitConfig, run_pipeline_multi_problem
from .run_ablation import (
    DEFAULT_HOLDOUT_SETS,
    DEFAULT_SEEDS,
    _release_gpu_memory,
    _short_config,
    _summarize_by_variant,
)

# Sparse, the current shipped default, and denser -- brackets the current
# default from both sides rather than only probing "more points."
DEFAULT_LHS_POINTS = [5, 20, 50]

DEFAULT_FEATURE_MODE = "lhs"


def run_one_lhs_points_variant(
    *,
    n_lhs_points: int,
    seed: int,
    holdout_fids: list[int],
    data_dir: str | Path,
    output_dir: Path,
    feature_mode: str,
    max_records: int,
    base_config_path: str | Path,
    max_epochs: int,
    max_steps_per_epoch: int,
    patience: int,
    include_baselines: bool,
    predict_batch_size: int,
) -> dict[str, Any]:
    from . import evaluate as evaluate_module
    from . import train as train_module

    holdout_tag = "-".join(str(f) for f in holdout_fids)
    run_dir = output_dir / f"lhs{n_lhs_points}__seed{seed}__holdout{holdout_tag}"
    data_out = run_dir / "data"
    checkpoint_dir = run_dir / "checkpoint"

    t0 = time.time()
    pipeline_summary = run_pipeline_multi_problem(
        data_dir,
        data_out,
        target="aucs_per_instance",
        feature_mode=feature_mode,
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

    # Same rationale as run_ablation.py's identical call: train_module.train()
    # builds its own RLM (model + optimizer state) that goes out of scope
    # here; without this, that GPU memory can still be resident/fragmented
    # when evaluate_module below loads a second RLM for evaluation.
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
        "variant": str(n_lhs_points),
        "n_lhs_points": n_lhs_points,
        "feature_mode": feature_mode,
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
    with open(run_dir / "ablation_result.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def run_lhs_points_ablation(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    lhs_points_variants: list[int] = DEFAULT_LHS_POINTS,
    seeds: list[int] = DEFAULT_SEEDS,
    holdout_sets: list[list[int]] = DEFAULT_HOLDOUT_SETS,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    max_records: int = 6000,
    base_config_path: str | Path = Path(__file__).parent / "configs" / "default.yaml",
    max_epochs: int = 20,
    max_steps_per_epoch: int = 300,
    patience: int = 6,
    include_baselines: bool = False,
    predict_batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Runs every ``(n_lhs_points, seed, holdout_fids)`` combination and
    writes a summary table (including a ``by_variant`` mean/std rollup, see
    ``run_ablation._summarize_by_variant``) to
    ``output_dir/ablation_summary.json``. Returns the list of per-run result
    dicts (also what gets written)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n_lhs_points in lhs_points_variants:
        for seed in seeds:
            for holdout_fids in holdout_sets:
                print(
                    f"=== n_lhs_points={n_lhs_points} seed={seed} "
                    f"holdout_fids={list(holdout_fids)} ==="
                )
                result = run_one_lhs_points_variant(
                    n_lhs_points=n_lhs_points,
                    seed=seed,
                    holdout_fids=list(holdout_fids),
                    data_dir=data_dir,
                    output_dir=output_dir,
                    feature_mode=feature_mode,
                    max_records=max_records,
                    base_config_path=base_config_path,
                    max_epochs=max_epochs,
                    max_steps_per_epoch=max_steps_per_epoch,
                    patience=patience,
                    include_baselines=include_baselines,
                    predict_batch_size=predict_batch_size,
                )
                results.append(result)
                # Same rationale as run_ablation.py: without this, one run's
                # leftover GPU memory can still be resident (and fragmented)
                # when the next (n_lhs_points, seed, holdout_fids) starts
                # training a fresh RLM.
                _release_gpu_memory()
                print(
                    f"  -> spearman={result['spearman_rho']:.3f} "
                    f"kendall={result['kendall_tau']:.3f} "
                    f"n_test={result['n_test']} "
                    f"({sum(result['wall_clock_seconds'].values()):.0f}s)"
                )

    with open(output_dir / "ablation_summary.json", "w") as fh:
        json.dump(
            {
                "lhs_points_variants": list(lhs_points_variants),
                "feature_mode": feature_mode,
                "holdout_sets": [list(h) for h in holdout_sets],
                "max_records": max_records,
                "seeds": list(seeds),
                "by_variant": _summarize_by_variant(results),
                "results": results,
            },
            fh,
            indent=2,
            default=str,
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
        "--lhs-points",
        type=int,
        nargs="+",
        default=DEFAULT_LHS_POINTS,
        dest="lhs_points_variants",
        help=f"LHS sample-count variants to compare. Default: "
        f"{DEFAULT_LHS_POINTS} (sparse, current shipped default, denser).",
    )
    p.add_argument(
        "--feature-mode",
        default=DEFAULT_FEATURE_MODE,
        choices=["lhs", "lhs_stats", "meta", "meta+lhs", "meta+lhs_stats"],
        help="Feature-mode text representation held fixed across all "
        f"n_lhs_points variants (default: {DEFAULT_FEATURE_MODE!r}). Only "
        "modes that draw an LHS sample (anything other than 'meta' alone) "
        "are actually sensitive to --lhs-points.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Repeat each (n_lhs_points, holdout_fids) combination for "
        f"each of these seeds. Default: {DEFAULT_SEEDS} (same as "
        "run_ablation.py, for comparability).",
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
        f"holdout set. Default (if omitted entirely): {DEFAULT_HOLDOUT_SETS} "
        "(same as run_ablation.py, for comparability).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=6000,
        help="Subsample size for the fast ablation pass -- not the full "
        "dataset. See run_ablation.py's identical flag for the wall-clock "
        "caveats around raising this.",
    )
    p.add_argument(
        "--base-config",
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
        dest="base_config_path",
        help="Model/optimizer config to start from; only max_epochs/"
        "max_steps_per_epoch/patience are overridden for the ablation "
        "budget.",
    )
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--max-steps-per-epoch", type=int, default=300)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also compute the feature/random baselines per run (slower, "
        "not needed just to rank n_lhs_points variants against each other).",
    )
    p.add_argument(
        "--predict-batch-size",
        type=int,
        default=4,
        help="Chunk size for RLM sampling-based prediction during "
        "evaluation -- see run_ablation.py's identical flag for the "
        "real-OOM background behind the conservative default.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    holdout_sets = (
        args.holdout_sets if args.holdout_sets is not None else DEFAULT_HOLDOUT_SETS
    )
    results = run_lhs_points_ablation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lhs_points_variants=args.lhs_points_variants,
        seeds=args.seeds,
        holdout_sets=holdout_sets,
        feature_mode=args.feature_mode,
        max_records=args.max_records,
        base_config_path=args.base_config_path,
        max_epochs=args.max_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        patience=args.patience,
        include_baselines=args.include_baselines,
        predict_batch_size=args.predict_batch_size,
    )
    print("\n=== Ablation summary (individual runs) ===")
    for r in sorted(results, key=lambda r: -r["spearman_rho"]):
        print(
            f"n_lhs_points={r['n_lhs_points']:<4} seed={r['seed']} "
            f"holdout={r['holdout_fids']} spearman={r['spearman_rho']:.3f} "
            f"kendall={r['kendall_tau']:.3f} n_test={r['n_test']}"
        )

    print("\n=== Ablation summary (by n_lhs_points, mean +/- std) ===")
    by_variant = _summarize_by_variant(results)
    for variant, stats in sorted(
        by_variant.items(), key=lambda kv: -kv[1]["spearman_mean"]
    ):
        print(
            f"n_lhs_points={variant:<4} n_runs={stats['n_runs']} "
            f"spearman={stats['spearman_mean']:.3f}+/-{stats['spearman_std']:.3f} "
            f"kendall={stats['kendall_mean']:.3f}+/-{stats['kendall_std']:.3f}"
        )


if __name__ == "__main__":
    main()
