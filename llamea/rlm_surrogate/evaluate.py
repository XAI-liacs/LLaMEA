"""Step 3 evaluation: mirrors the RLM paper's diagnostics rather than a
single correlation number.

Pure metric functions (``rank_metrics``, ``within_run_ranking``,
``topk_overlap``, ``budget_reduction_simulation``) only need numpy/scipy and
operate on plain (true, predicted) arrays, so they're testable without
torch/regress_lm installed. Model loading and prediction (``load_trained_rlm``,
``predict_with_rlm``) import torch/regress_lm lazily.

CLI:
    python -m llamea.rlm_surrogate.evaluate \\
        --checkpoint-dir checkpoints/run1 --train data/train.jsonl \\
        --test data/test.jsonl --output-dir results/eval_run1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats as scipy_stats

from .data_pipeline import RLMExample, read_examples_jsonl

# --------------------------------------------------------------------------
# Pure metrics.
# --------------------------------------------------------------------------


def rank_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Spearman-rho and Kendall-tau between predicted and true performance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return {
            "n": len(y_true),
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
        }
    rho, _ = scipy_stats.spearmanr(y_true, y_pred)
    tau, _ = scipy_stats.kendalltau(y_true, y_pred)
    return {"n": len(y_true), "spearman_rho": float(rho), "kendall_tau": float(tau)}


def topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Fraction of the true top-k also captured in the predicted top-k."""
    n = len(y_true)
    k = max(1, min(k, n))
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    return len(true_top & pred_top) / k


def random_topk_overlap_expected(k: int, n: int) -> float:
    """Expected top-k overlap of a uniformly random selection (hypergeometric mean)."""
    k = max(1, min(k, n))
    return k / n


def within_run_ranking(
    examples: Sequence[RLMExample],
    y_pred: Sequence[float],
    *,
    min_n: int = 5,
    top_p_list: Sequence[float] = (0.1, 0.25, 0.5),
) -> dict[str, Any]:
    """Per-run Spearman-rho and top-p% selection accuracy vs. random,
    analogous to the paper's Figure 5 / Figure 11."""
    by_run: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        by_run[e.run_id].append(i)

    y_true_all = np.array([e.fitness for e in examples], dtype=float)
    y_pred_all = np.asarray(y_pred, dtype=float)

    per_run: dict[str, Any] = {}
    for run_id, idxs in by_run.items():
        if len(idxs) < min_n:
            continue
        yt = y_true_all[idxs]
        yp = y_pred_all[idxs]
        n = len(idxs)
        entry: dict[str, Any] = {"n": n, **rank_metrics(yt, yp)}
        entry["top1_hit"] = bool(np.argmax(yp) == np.argmax(yt))
        entry["topk"] = {}
        for p in top_p_list:
            k = max(1, round(p * n))
            overlap = topk_overlap(yt, yp, k)
            entry["topk"][f"top_{int(p * 100)}pct"] = {
                "k": k,
                "rlm_overlap": overlap,
                "random_expected_overlap": random_topk_overlap_expected(k, n),
            }
        per_run[run_id] = entry

    if not per_run:
        return {"per_run": {}, "n_runs_evaluated": 0}

    agg_rho = [
        v["spearman_rho"] for v in per_run.values() if np.isfinite(v["spearman_rho"])
    ]
    agg_top1 = [v["top1_hit"] for v in per_run.values()]
    return {
        "per_run": per_run,
        "n_runs_evaluated": len(per_run),
        "mean_spearman_rho": float(np.mean(agg_rho)) if agg_rho else float("nan"),
        "top1_selection_accuracy": float(np.mean(agg_top1)),
    }


@dataclass
class BudgetSimEntry:
    k_over_n: float
    n_batches: int
    mean_k: float
    rlm_best_found_fraction: float  # of the true best-in-batch, on average
    random_best_found_fraction: float
    all_best_found_fraction: float  # always 1.0; included for symmetry/sanity
    rlm_regret: float  # true_best - best_found_by_rlm_selection, averaged
    random_regret: float
    compute_savings: float  # 1 - k/N


def budget_reduction_simulation(
    examples: Sequence[RLMExample],
    y_pred: Sequence[float],
    *,
    k_over_n_list: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    min_batch_size: int = 4,
    n_random_draws: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Given a generation of N candidates (grouped by (run_id, generation)),
    simulates "evaluate only the top-k predicted by the RLM" vs "evaluate k
    random candidates" vs "evaluate all N", reporting how close the best-found
    performance gets and the resulting compute savings."""
    rng = np.random.default_rng(seed)

    by_batch: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        by_batch[(e.run_id, e.generation)].append(i)
    batches = {
        key: idxs for key, idxs in by_batch.items() if len(idxs) >= min_batch_size
    }

    y_true_all = np.array([e.fitness for e in examples], dtype=float)
    y_pred_all = np.asarray(y_pred, dtype=float)

    results = []
    for k_over_n in k_over_n_list:
        rlm_fracs, random_fracs, rlm_regrets, random_regrets, ks = [], [], [], [], []
        for idxs in batches.values():
            idxs = np.array(idxs)
            n = len(idxs)
            k = max(1, round(k_over_n * n))
            ks.append(k)
            yt = y_true_all[idxs]
            yp = y_pred_all[idxs]
            true_best = yt.max()
            if true_best == 0:
                continue

            rlm_selected = np.argsort(-yp)[:k]
            rlm_best = yt[rlm_selected].max()
            rlm_fracs.append(rlm_best / true_best)
            rlm_regrets.append(true_best - rlm_best)

            draw_bests = []
            for _ in range(n_random_draws):
                draw = rng.choice(n, size=k, replace=False)
                draw_bests.append(yt[draw].max())
            random_best = float(np.mean(draw_bests))
            random_fracs.append(random_best / true_best)
            random_regrets.append(true_best - random_best)

        if not rlm_fracs:
            continue
        results.append(
            BudgetSimEntry(
                k_over_n=k_over_n,
                n_batches=len(rlm_fracs),
                mean_k=float(np.mean(ks)),
                rlm_best_found_fraction=float(np.mean(rlm_fracs)),
                random_best_found_fraction=float(np.mean(random_fracs)),
                all_best_found_fraction=1.0,
                rlm_regret=float(np.mean(rlm_regrets)),
                random_regret=float(np.mean(random_regrets)),
                compute_savings=1.0 - k_over_n,
            )
        )

    return {
        "n_batches_total": len(batches),
        "min_batch_size": min_batch_size,
        "results": [asdict(r) for r in results],
    }


def evaluate_predictions(
    examples: Sequence[RLMExample],
    y_pred: Sequence[float],
    *,
    label: str = "",
    min_n_within_run: int = 5,
    top_p_list: Sequence[float] = (0.1, 0.25, 0.5),
    k_over_n_list: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    y_true = [e.fitness for e in examples]
    overall = rank_metrics(y_true, y_pred)
    by_run = {}
    run_ids = sorted({e.run_id for e in examples})
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    for run_id in run_ids:
        mask = [e.run_id == run_id for e in examples]
        by_run[run_id] = rank_metrics(y_true_arr[mask], y_pred_arr[mask])

    by_problem = {}
    problem_ids = sorted({e.problem_id for e in examples if e.problem_id})
    for problem_id in problem_ids:
        mask = [e.problem_id == problem_id for e in examples]
        by_problem[problem_id] = rank_metrics(y_true_arr[mask], y_pred_arr[mask])

    return {
        "label": label,
        "overall": overall,
        "by_run": by_run,
        "by_problem": by_problem,
        "within_run_ranking": within_run_ranking(
            examples, y_pred, min_n=min_n_within_run, top_p_list=top_p_list
        ),
        "budget_reduction": budget_reduction_simulation(
            examples, y_pred, k_over_n_list=k_over_n_list
        ),
    }


# --------------------------------------------------------------------------
# Model loading / prediction (torch/regress_lm imported lazily).
# --------------------------------------------------------------------------


def load_trained_rlm(checkpoint_dir: str | Path):
    import torch

    from .config import RLMSurrogateConfig
    from .model import build_model

    checkpoint_dir = Path(checkpoint_dir)
    config = RLMSurrogateConfig.from_yaml(checkpoint_dir / "config.yaml")
    reg_lm = build_model(config)
    state = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    reg_lm.model.load_state_dict(state)
    reg_lm.model.eval()
    return reg_lm, config


def predict_with_rlm(reg_lm, config, examples: Sequence[RLMExample]) -> np.ndarray:
    from .model import median_point_predictions, scalar_point_predictions

    medians = median_point_predictions(
        reg_lm, [e.x for e in examples], num_samples=config.num_samples_point_pred
    )
    return scalar_point_predictions(medians)


def predict_with_feature_baseline(
    train_examples: Sequence[RLMExample],
    eval_examples: Sequence[RLMExample],
    seed: int = 0,
) -> np.ndarray:
    from .features import FeatureMatrixBuilder, XGBFeatureBaseline

    builder = FeatureMatrixBuilder()
    X_train = builder.fit_transform(train_examples)
    X_eval = builder.transform(eval_examples)
    y_train = np.array([e.fitness for e in train_examples])
    model = XGBFeatureBaseline(seed=seed).fit(X_train, y_train)
    return model.predict(X_eval)


def predict_random(examples: Sequence[RLMExample], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(size=len(examples))


def aggregate_instance_predictions(
    examples: Sequence[RLMExample], y_pred: Sequence[float]
) -> tuple[list[RLMExample], np.ndarray]:
    """Rolls per-instance predictions (from `target="aucs_per_instance"`
    exploded examples, one row per `metadata.aucs[i]`) back up to one
    prediction per candidate, by averaging `y_pred` across each
    candidate's exploded rows -- this is what "which candidate to
    evaluate next" decisions (within-run ranking, budget-reduction sim)
    actually need. No-op passthrough (returns inputs unchanged) when
    `examples` weren't exploded (every `candidate_id` empty).

    Ground truth for the aggregated view is each candidate's own
    `fitness` field (already the true mean of `aucs`), not re-derived
    from `y_true`.
    """
    if not any(e.candidate_id for e in examples):
        return list(examples), np.asarray(y_pred, dtype=float)

    y_pred_arr = np.asarray(y_pred, dtype=float)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        groups[e.candidate_id or e.id].append(i)

    agg_examples = []
    agg_pred = []
    for candidate_id, idxs in groups.items():
        representative = examples[idxs[0]]
        agg_examples.append(
            replace(representative, id=candidate_id, candidate_id="", instance_index=-1)
        )
        agg_pred.append(float(np.mean(y_pred_arr[idxs])))
    return agg_examples, np.array(agg_pred)


def run_full_evaluation(
    checkpoint_dir: str | Path,
    train_path: str | Path,
    eval_path: str | Path,
    *,
    include_baselines: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    train_examples = read_examples_jsonl(train_path)
    eval_examples = read_examples_jsonl(eval_path)
    is_exploded = any(e.candidate_id for e in eval_examples)

    reg_lm, config = load_trained_rlm(checkpoint_dir)
    rlm_pred_raw = predict_with_rlm(reg_lm, config, eval_examples)
    eval_examples_agg, rlm_pred = aggregate_instance_predictions(
        eval_examples, rlm_pred_raw
    )

    report: dict[str, Any] = {
        "n_train": len(train_examples),
        "n_eval": len(eval_examples_agg),
        "checkpoint_dir": str(checkpoint_dir),
        "arms": {
            "rlm": evaluate_predictions(eval_examples_agg, rlm_pred, label="rlm"),
        },
        "predictions": {"rlm": _predictions_payload(eval_examples_agg, rlm_pred)},
    }
    if is_exploded:
        report["instance_level"] = {
            "rlm": rank_metrics([e.y for e in eval_examples], rlm_pred_raw)
        }

    if include_baselines:
        feature_pred_raw = predict_with_feature_baseline(
            train_examples, eval_examples, seed=seed
        )
        random_pred_raw = predict_random(eval_examples, seed=seed)
        _, feature_pred = aggregate_instance_predictions(
            eval_examples, feature_pred_raw
        )
        _, random_pred = aggregate_instance_predictions(eval_examples, random_pred_raw)

        report["arms"]["feature_baseline"] = evaluate_predictions(
            eval_examples_agg, feature_pred, label="feature_baseline"
        )
        report["arms"]["random"] = evaluate_predictions(
            eval_examples_agg, random_pred, label="random"
        )
        report["predictions"]["feature_baseline"] = _predictions_payload(
            eval_examples_agg, feature_pred
        )
        report["predictions"]["random"] = _predictions_payload(
            eval_examples_agg, random_pred
        )
        if is_exploded:
            report["instance_level"]["feature_baseline"] = rank_metrics(
                [e.y for e in eval_examples], feature_pred_raw
            )
            report["instance_level"]["random"] = rank_metrics(
                [e.y for e in eval_examples], random_pred_raw
            )

    return report


def _predictions_payload(
    examples: Sequence[RLMExample], y_pred: Sequence[float]
) -> dict[str, Any]:
    return {
        "id": [e.id for e in examples],
        "run_id": [e.run_id for e in examples],
        "problem_id": [e.problem_id for e in examples],
        "generation": [e.generation for e in examples],
        "y_true": [e.fitness for e in examples],
        "y_pred": [float(v) for v in y_pred],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument(
        "--train", required=True, help="Train split (for the feature baseline)."
    )
    p.add_argument("--test", required=True, help="Held-out split to evaluate on.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-baselines", dest="include_baselines", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(include_baselines=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    report = run_full_evaluation(
        args.checkpoint_dir,
        args.train,
        args.test,
        include_baselines=args.include_baselines,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    rlm_overall = report["arms"]["rlm"]["overall"]
    print(
        f"RLM overall: n={rlm_overall['n']} spearman={rlm_overall['spearman_rho']:.3f} "
        f"kendall={rlm_overall['kendall_tau']:.3f}"
    )
    if "feature_baseline" in report["arms"]:
        fb = report["arms"]["feature_baseline"]["overall"]
        print(f"Feature baseline: spearman={fb['spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
