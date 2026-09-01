"""Problem-feature ablation: trains + evaluates the RLM surrogate under a
few different "# Problem" feature representations (see
``problem_instances.compute_problem_feature_text``'s ``mode`` menu) on a
*subsampled* dataset with a *short* training budget, so you can rank
variants in a reasonable time before committing the full compute budget to
whichever wins.

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

Requires the ``ioh`` extra, real ``BLADE-results`` data in the
``per_problem_subdir`` layout, and a GPU for the T5Gemma config -- this is a
driver, not something exercised in the (CPU-only, synthetic-fixture) test
suite. To add more seeds per variant, pass ``--seeds 0 1 2``; to try
additional variants, pass ``--variants`` with any subset of
``lhs lhs_stats meta meta+lhs meta+lhs_stats``.

CLI:
    uv run python -m llamea.rlm_surrogate.run_ablation \\
        --data-dir /data/BLADE-results --output-dir results/ablation \\
        --holdout-fids 21 22 --max-records 6000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import RLMSurrogateConfig
from .data_pipeline import SplitConfig, run_pipeline_multi_problem

DEFAULT_VARIANTS = ["lhs", "meta+lhs", "meta+lhs_stats"]


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
) -> dict[str, Any]:
    from . import evaluate as evaluate_module
    from . import train as train_module

    run_dir = output_dir / f"{variant.replace('+', '_')}__seed{seed}"
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

    t0 = time.time()
    eval_report = evaluate_module.run_full_evaluation(
        checkpoint_dir,
        data_out / "train.jsonl",
        data_out / "test.jsonl",
        include_baselines=include_baselines,
        seed=seed,
    )
    t_eval = time.time() - t0

    rlm_overall = eval_report["arms"]["rlm"]["overall"]
    result = {
        "variant": variant,
        "seed": seed,
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


def run_ablation(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    variants: list[str] = DEFAULT_VARIANTS,
    seeds: list[int] = (0,),
    holdout_fids: list[int] = (21, 22),
    max_records: int = 6000,
    n_lhs_points: int = 20,
    base_config_path: str | Path = Path(__file__).parent / "configs" / "default.yaml",
    max_epochs: int = 10,
    max_steps_per_epoch: int = 200,
    patience: int = 4,
    include_baselines: bool = False,
) -> list[dict[str, Any]]:
    """Runs every ``(variant, seed)`` combination and writes a summary
    table to ``output_dir/ablation_summary.json``. Returns the list of
    per-run result dicts (also what gets written)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in variants:
        for seed in seeds:
            print(f"=== variant={variant!r} seed={seed} ===")
            result = run_one_variant(
                variant=variant,
                seed=seed,
                data_dir=data_dir,
                output_dir=output_dir,
                holdout_fids=list(holdout_fids),
                max_records=max_records,
                n_lhs_points=n_lhs_points,
                base_config_path=base_config_path,
                max_epochs=max_epochs,
                max_steps_per_epoch=max_steps_per_epoch,
                patience=patience,
                include_baselines=include_baselines,
            )
            results.append(result)
            print(
                f"  -> spearman={result['spearman_rho']:.3f} "
                f"kendall={result['kendall_tau']:.3f} "
                f"n_test={result['n_test']} "
                f"({sum(result['wall_clock_seconds'].values()):.0f}s)"
            )

    with open(output_dir / "ablation_summary.json", "w") as fh:
        json.dump(
            {
                "holdout_fids": list(holdout_fids),
                "max_records": max_records,
                "variants": variants,
                "seeds": list(seeds),
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
        default=[0],
        help="Repeat each variant for each of these seeds. Default: a "
        "single seed (matches the '3 runs' minimal ablation); add more "
        "for statistical confidence once the harness is confirmed working.",
    )
    p.add_argument(
        "--holdout-fids",
        type=int,
        nargs="+",
        default=[21, 22],
        help="BBOB function ids held out entirely for test (leave-function-"
        "out split). Default: f21/f22 (Gallagher's Gaussian Peaks -- "
        "distinctive multi-modal, weak-global-structure landscapes).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=6000,
        help="Subsample size for the fast ablation pass -- not the full "
        "dataset. Raise once you're ready for a less noisy comparison.",
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
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--max-steps-per-epoch", type=int, default=200)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also compute the feature/random baselines per run (slower, "
        "not needed just to rank variants against each other).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    results = run_ablation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        variants=args.variants,
        seeds=args.seeds,
        holdout_fids=args.holdout_fids,
        max_records=args.max_records,
        n_lhs_points=args.n_lhs_points,
        base_config_path=args.base_config_path,
        max_epochs=args.max_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        patience=args.patience,
        include_baselines=args.include_baselines,
    )
    print("\n=== Ablation summary ===")
    for r in sorted(results, key=lambda r: -r["spearman_rho"]):
        print(
            f"{r['variant']:<16} seed={r['seed']} "
            f"spearman={r['spearman_rho']:.3f} kendall={r['kendall_tau']:.3f} "
            f"n_test={r['n_test']}"
        )


if __name__ == "__main__":
    main()
