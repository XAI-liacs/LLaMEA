"""Step 4: assembles ``results/report.md`` from the data-pipeline stats and
evaluate.py's output, with matplotlib plots (predicted-vs-true scatter,
within-run Spearman-rho by run, and the budget-reduction curve).

CLI:
    python -m llamea.rlm_surrogate.report \\
        --stats data/stats.json --eval-results results/eval_run1/eval_results.json \\
        --config checkpoints/run1/config.yaml --output-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _make_plots(eval_results: dict[str, Any], figures_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    preds = eval_results.get("predictions", {})
    if "rlm" in preds:
        rlm = preds["rlm"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(rlm["y_true"], rlm["y_pred"], alpha=0.6, s=18)
        lo = min(min(rlm["y_true"]), min(rlm["y_pred"]))
        hi = max(max(rlm["y_true"]), max(rlm["y_pred"]))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax.set_xlabel("True fitness")
        ax.set_ylabel("RLM predicted fitness (median-of-N)")
        ax.set_title("RLM: predicted vs. true (held-out)")
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "predicted_vs_true.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["predicted_vs_true"] = str(path)

    within_run = (
        eval_results.get("arms", {}).get("rlm", {}).get("within_run_ranking", {})
    )
    per_run = within_run.get("per_run", {})
    if per_run:
        runs = list(per_run.keys())
        rhos = [per_run[r]["spearman_rho"] for r in runs]
        fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(runs)), 4))
        ax.bar(runs, rhos)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Within-run Spearman-rho")
        ax.set_title("RLM ranking quality by run")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        path = figures_dir / "within_run_spearman.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["within_run_spearman"] = str(path)

    budget = eval_results.get("arms", {}).get("rlm", {}).get("budget_reduction", {})
    budget_random = (
        eval_results.get("arms", {}).get("random", {}).get("budget_reduction", {})
    )
    results = budget.get("results", [])
    if results:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        k_over_n = [r["k_over_n"] for r in results]
        rlm_frac = [r["rlm_best_found_fraction"] for r in results]
        random_frac = [r["random_best_found_fraction"] for r in results]
        ax.plot(k_over_n, rlm_frac, "o-", label="RLM-selected top-k")
        ax.plot(k_over_n, random_frac, "s--", label="Random-selected k")
        ax.axhline(
            1.0, color="gray", linewidth=0.8, linestyle=":", label="Evaluate all N"
        )
        ax.set_xlabel("k / N evaluated")
        ax.set_ylabel("Fraction of best-in-batch fitness found")
        ax.set_title("Budget-reduction simulation")
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "budget_reduction.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["budget_reduction"] = str(path)

    return paths


def _by_problem_table(arms: dict[str, Any]) -> str:
    problem_ids: set[str] = set()
    for arm in arms.values():
        problem_ids.update(arm.get("by_problem", {}).keys())
    if not problem_ids:
        return "_No `problem_id` grouping available for this dataset (flat layout)._\n"

    lines = [
        "| Problem | " + " | ".join(f"{name} (n / rho / tau)" for name in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for problem_id in sorted(problem_ids):
        cells = []
        for arm in arms.values():
            m = arm.get("by_problem", {}).get(problem_id)
            cells.append(
                f"{m['n']} / {_fmt(m['spearman_rho'])} / {_fmt(m['kendall_tau'])}"
                if m
                else "n/a"
            )
        lines.append(f"| {problem_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _within_run_table(within_run: dict[str, Any]) -> str:
    per_run = within_run.get("per_run", {})
    if not per_run:
        return "_No runs had enough held-out candidates for within-run evaluation._\n"
    lines = [
        "| Run | n | Spearman-rho | Top-1 hit | Top-25% overlap (RLM / random-expected) |",
        "|---|---|---|---|---|",
    ]
    for run_id, v in sorted(per_run.items()):
        top25 = v.get("topk", {}).get("top_25pct", {})
        lines.append(
            f"| {run_id} | {v['n']} | {_fmt(v['spearman_rho'])} | "
            f"{'yes' if v['top1_hit'] else 'no'} | "
            f"{_fmt(top25.get('rlm_overlap'))} / {_fmt(top25.get('random_expected_overlap'))} |"
        )
    return "\n".join(lines) + "\n"


def _budget_table(budget: dict[str, Any]) -> str:
    results = budget.get("results", [])
    if not results:
        return "_No batches (run, generation) met the minimum batch size for simulation._\n"
    lines = [
        "| k/N | mean k | batches | RLM best-found frac. | Random best-found frac. | Compute savings |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['k_over_n']:.2f} | {r['mean_k']:.1f} | {r['n_batches']} | "
            f"{_fmt(r['rlm_best_found_fraction'])} | {_fmt(r['random_best_found_fraction'])} | "
            f"{r['compute_savings']:.0%} |"
        )
    return "\n".join(lines) + "\n"


def build_report(
    stats: dict[str, Any],
    eval_results: dict[str, Any],
    model_config: dict[str, Any],
    figures: dict[str, str],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# LLaMEA RLM surrogate: data, model, and evaluation report\n")

    # --- Data ---
    lines.append("## 1. Data\n")
    is_multi_problem = stats.get("layout") == "per_problem_subdir"
    n_sources = (
        len(stats.get("problems", [])) if is_multi_problem else stats.get("n_files")
    )
    source_label = "problem(s)" if is_multi_problem else "file(s)/run(s)"
    lines.append(
        f"Ingested **{stats['n_records_total']}** records "
        f"({'problems: ' + ', '.join(stats.get('problems', [])) if is_multi_problem else f'from **{n_sources}** {source_label}'})"
        f". Dropped **{stats['n_errored_total']}** "
        f"({stats['error_fraction_total']:.1%}) as invalid -- a non-empty `error` field "
        "OR a non-finite (`inf`/`-inf`/`nan`) `fitness` with an empty `error` (some "
        "logger versions never populate `error` on failure; see `BladeRecord.is_invalid`). "
        f"**{stats['n_examples_total']}** usable `(x, y)` examples remain "
        f"(target = `{stats['target']}`, description included = "
        f"{stats['include_description']}, configspace included = "
        f"{stats['include_configspace']}).\n"
    )
    if is_multi_problem and stats.get("excluded_experiment_folders"):
        lines.append(
            '> **Excluded folders** (name contained "debug"/"wrong", treated as '
            "scratch/known-bad runs, not real experiments): "
            f"{', '.join(stats['excluded_experiment_folders'])}. Rerun "
            "`data_pipeline.py --exclude-dir-substring ''` to include everything.\n"
        )
    lines.append("### Per-run summary\n")
    lines.append(
        "| File | n | Errored (frac) | fitness min/max/mean | fitness skew | "
        "gen-vs-fitness Spearman | warnings |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for f in stats["per_file"]:
        warn = "; ".join(f["warnings"]) or "-"
        lines.append(
            f"| {f['file']} | {f['n_records']} | "
            f"{f['n_errored']} ({f['error_fraction']:.1%}) | "
            f"{_fmt(f['fitness_min'])}/{_fmt(f['fitness_max'])}/{_fmt(f['fitness_mean'])} | "
            f"{_fmt(f['fitness_skew'])} | {_fmt(f['generation_fitness_spearman'])} | {warn} |"
        )
    lines.append("")
    for w in stats.get("warnings", []):
        lines.append(f"> **Warning:** {w}\n")

    # --- Split ---
    lines.append("## 2. Train/val/test split\n")
    split = stats["split"]
    lines.append(
        f"Strategy: **{split['strategy']}**. This is deliberately *not* a "
        "random i.i.d. split: whole runs are held out for the test set "
        "whenever enough runs are available, and otherwise (or for the "
        "validation set) later generations within a run are held out rather "
        "than randomly interspersed, so near-duplicate mutated siblings of a "
        "training example can't leak into val/test.\n"
    )
    lines.append(
        f"- train: {split['n_train']}, val: {split['n_val']}, test: {split['n_test']}"
    )
    if split.get("test_runs"):
        lines.append(
            f"- runs held out entirely for test: {', '.join(split['test_runs'])}"
        )
    if split.get("val_generation_cutoffs"):
        lines.append(
            f"- per-run generation cutoff used for val: `{split['val_generation_cutoffs']}`"
        )
    if split.get("test_generation_cutoffs"):
        lines.append(
            f"- per-run generation cutoff used for test: `{split['test_generation_cutoffs']}`"
        )
    if split.get("by_problem"):
        lines.append(
            "\nThe split was run **independently per problem** and then merged, so "
            "one problem's runs can't dominate the held-out set:\n"
        )
        lines.append("| Problem | n_runs | strategy | train | val | test |")
        lines.append("|---|---|---|---|---|---|")
        for problem_id, log in sorted(split["by_problem"].items()):
            lines.append(
                f"| {problem_id} | {log.get('n_runs', 'n/a')} | {log.get('strategy')} | "
                f"{log.get('n_train')} | {log.get('n_val')} | {log.get('n_test')} |"
            )
    lines.append("")

    # --- Model ---
    lines.append("## 3. Model configuration\n")
    lines.append("```yaml")
    import yaml

    lines.append(yaml.safe_dump(model_config, sort_keys=False).strip())
    lines.append("```\n")

    # --- Metrics ---
    lines.append("## 4. Evaluation\n")
    arms = eval_results.get("arms", {})
    lines.append(
        f"Evaluated on **{eval_results.get('n_eval')}** held-out examples "
        f"(feature baseline fit on **{eval_results.get('n_train')}** train examples).\n"
    )
    lines.append("### 4.1 Overall ranking quality\n")
    lines.append("| Arm | n | Spearman-rho | Kendall-tau |")
    lines.append("|---|---|---|---|")
    for name, arm in arms.items():
        o = arm["overall"]
        lines.append(
            f"| {name} | {o['n']} | {_fmt(o['spearman_rho'])} | {_fmt(o['kendall_tau'])} |"
        )
    lines.append("")

    if "predicted_vs_true" in figures:
        lines.append(
            f"![Predicted vs true]({Path(figures['predicted_vs_true']).name})\n"
        )

    lines.append("### 4.2 Ranking quality by benchmark problem\n")
    lines.append(_by_problem_table(arms))

    lines.append("### 4.3 Within-run ranking and selection accuracy (RLM)\n")
    lines.append(
        "This is the number that most directly answers whether pre-screening "
        "helps: for each run with enough held-out candidates, the "
        "within-run Spearman-rho and whether the model's top pick is the "
        "actual best-performing candidate in that run.\n"
    )
    lines.append(_within_run_table(arms.get("rlm", {}).get("within_run_ranking", {})))
    if "within_run_spearman" in figures:
        lines.append(
            f"\n![Within-run Spearman]({Path(figures['within_run_spearman']).name})\n"
        )

    lines.append("### 4.4 Budget-reduction simulation\n")
    lines.append(
        "Given a generation of N candidates, evaluating only the top-k "
        "predicted by the RLM vs. k random candidates vs. all N:\n"
    )
    lines.append(_budget_table(arms.get("rlm", {}).get("budget_reduction", {})))
    if "budget_reduction" in figures:
        lines.append(
            f"\n![Budget reduction]({Path(figures['budget_reduction']).name})\n"
        )

    # --- Recommendation ---
    lines.append("## 5. Recommendation\n")
    lines.append(_recommendation(stats, eval_results))

    report = "\n".join(lines)
    with open(output_dir / "report.md", "w") as fh:
        fh.write(report)
    return report


def _recommendation(stats: dict[str, Any], eval_results: dict[str, Any]) -> str:
    n_examples = stats["n_examples_total"]
    rlm_overall = eval_results.get("arms", {}).get("rlm", {}).get("overall", {})
    rho = rlm_overall.get("spearman_rho", float("nan"))
    within_run = (
        eval_results.get("arms", {}).get("rlm", {}).get("within_run_ranking", {})
    )
    n_runs_eval = within_run.get("n_runs_evaluated", 0)

    parts = []
    if n_examples < 1000:
        parts.append(
            f"**Sample size caveat:** only {n_examples} usable examples were used here. The "
            "RLM paper's strongest results come from tens-of-thousands to millions of "
            "examples; with a dataset this small, treat the correlation numbers above as "
            "a few-shot/fine-tuning signal, not a validated model. **Do not deploy this as "
            "a pre-screening gate until it has been re-evaluated on a pooled dataset of "
            "at least several hundred to low thousands of examples across multiple runs.**"
        )
    if n_runs_eval < 3:
        parts.append(
            f"Only {n_runs_eval} run(s) had enough held-out candidates for within-run "
            "evaluation -- the budget-reduction numbers above should be treated as "
            "illustrative of the *method*, not as a reliable estimate of savings on a "
            "new run, until more runs are pooled."
        )
    if np.isfinite(rho) and rho >= 0.5:
        parts.append(
            "Overall Spearman-rho is in a range where top-k pre-screening is plausibly "
            "useful. Suggested starting point: evaluate only the top ~50% of each "
            "generation by predicted score (k/N ~ 0.5), and cross-check against the "
            "budget-reduction table's `rlm_best_found_fraction` for that ratio before "
            "committing to a more aggressive cut."
        )
    elif np.isfinite(rho):
        parts.append(
            f"Overall Spearman-rho ({rho:.2f}) is too low to recommend pre-screening at "
            "this dataset size/model configuration. Retrain with more pooled data and/or "
            "the real frozen T5Gemma encoder (this run may have used the lightweight "
            "from-scratch/offline fallback encoder) before reconsidering."
        )
    parts.append(
        "Per-run fine-tuning (Step 2's `finetune_from_pool.yaml`) is worth testing once a "
        "pool of at least a few hundred historical examples across multiple runs exists; "
        "its value can't be assessed from a single run's data alone."
    )
    has_problem_breakdown = any(
        arm.get("by_problem") for arm in eval_results.get("arms", {}).values()
    )
    if not has_problem_breakdown:
        parts.append(
            "No `problem_id` grouping was available for this dataset (flat single-"
            "directory layout), so ranking quality could only be broken out by run, "
            "not by benchmark problem. Use `--layout per_problem_subdir` if the data "
            "has one experiment-folder per benchmark problem."
        )
    return "\n\n".join(f"- {p}" for p in parts) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats", required=True, help="stats.json from data_pipeline.py")
    p.add_argument(
        "--eval-results", required=True, help="eval_results.json from evaluate.py"
    )
    p.add_argument(
        "--config", required=True, help="config.yaml from the trained checkpoint"
    )
    p.add_argument("--output-dir", default="results")
    return p


def main(argv: list[str] | None = None) -> None:
    import yaml

    args = _build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.stats).read_text())
    eval_results = json.loads(Path(args.eval_results).read_text())
    model_config = yaml.safe_load(Path(args.config).read_text())

    figures = _make_plots(eval_results, output_dir / "figures")
    build_report(stats, eval_results, model_config, figures, output_dir)
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
