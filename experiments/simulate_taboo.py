#!/usr/bin/env python3
"""Taboo-search simulation over recorded LLaMEA-HPO run files.

Loads JSONL files produced by LLaMEA-HPO and replays each solution sequence
to measure how different taboo configurations would have filtered solutions
*before evaluation*.  Since the actual replacement candidates are not known,
this is a proxy analysis that answers two questions:

  1. **Pruning rate** – what fraction of solutions would be rejected?
  2. **Quality of pruning** – are the rejected solutions the bad ones?
     Measured as precision (of pruned, how many were below-median fitness)
     and recall (of all below-median solutions, how many were pruned).

Supported JSONL formats
-----------------------
* LLaMEA-HPO format : ``_solution`` (code), ``_fitness``, ``_generation``
* LLaMEA log format : ``code``, ``fitness``, ``generation``

Both can be mixed freely in the same data directory.

Usage
-----
    uv run python experiments/simulate_taboo.py --data-dir /path/to/jsonls
    uv run python experiments/simulate_taboo.py --data-dir /path/to/jsonls \\
        --thresholds 0.05 0.1 0.2 --out results.csv

Output
------
A CSV file (default: taboo_simulation_results.csv) and a summary table
printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llamea.utils import code_distance

# ---------------------------------------------------------------------------
# JSONL loading – handles both field-name conventions
# ---------------------------------------------------------------------------

def _get(record: dict, *keys, default=float("nan")):
    for k in keys:
        if k in record:
            return record[k]
    return default


def load_run(path: Path) -> list[dict]:
    """Return records from a JSONL file sorted by generation index."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(
                {
                    "code": _get(r, "_solution", "code", default=""),
                    "fitness": float(_get(r, "_fitness", "fitness", default=float("nan"))),
                    "generation": int(_get(r, "_generation", "generation", default=0)),
                    "name": _get(r, "_name", "name", default=""),
                }
            )
    records.sort(key=lambda r: r["generation"])
    return records


# ---------------------------------------------------------------------------
# Pairwise distance matrix (computed once per file, reused across all configs)
# ---------------------------------------------------------------------------

def build_distance_matrix(records: list[dict], verbose: bool = False) -> np.ndarray:
    """Return the n×n symmetric matrix of AST-based code distances."""
    n = len(records)
    D = np.zeros((n, n))
    pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = code_distance(records[i]["code"], records[j]["code"])
            D[i, j] = D[j, i] = d
            done += 1
            if verbose and done % max(1, pairs // 10) == 0:
                pct = 100 * done // pairs
                print(f"  distances {pct}% ({done}/{pairs})", end="\r", flush=True)
    if verbose:
        print()
    return D


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------

def build_config_grid(
    thresholds: list[float],
    stagnation_windows: list[int],
    poor_fitness_percentiles: list[float],
) -> list[dict]:
    configs: list[dict] = []

    for t in thresholds:
        configs.append({"strategy": "always", "threshold": t})

    for t in thresholds:
        for w in stagnation_windows:
            configs.append({"strategy": "stagnation", "threshold": t, "stagnation_window": w})

    for t in thresholds:
        for p in poor_fitness_percentiles:
            configs.append(
                {"strategy": "poor_fitness", "threshold": t, "poor_fitness_percentile": p}
            )

    return configs


def config_label(cfg: dict) -> str:
    s = cfg["strategy"]
    t = cfg["threshold"]
    if s == "always":
        return f"always(t={t})"
    if s == "stagnation":
        return f"stagnation(t={t},w={cfg['stagnation_window']})"
    if s == "poor_fitness":
        return f"poor_fitness(t={t},p={cfg['poor_fitness_percentile']})"
    return str(cfg)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _is_taboo(idx: int, taboo_indices: list[int], D: np.ndarray, threshold: float) -> bool:
    for ti in taboo_indices:
        if D[idx, ti] < threshold:
            return True
    return False


def simulate_run(
    records: list[dict],
    D: np.ndarray,
    config: dict,
    minimization: bool = False,
) -> pd.DataFrame:
    """Replay one run with one taboo configuration.

    Returns a DataFrame with one row per solution and columns:
    ``generation``, ``fitness``, ``is_taboo``.

    The taboo list is populated *after* each generation's check (mirroring the
    actual algorithm where filtering happens before evaluation and the list is
    updated afterwards).
    """
    strategy = config["strategy"]
    threshold = config["threshold"]
    stagnation_window = config.get("stagnation_window", 5)
    poor_pct = config.get("poor_fitness_percentile", 25.0)

    fitnesses = [r["fitness"] for r in records]

    # Group indices by generation
    gen_groups: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        gen_groups.setdefault(r["generation"], []).append(i)

    taboo_indices: list[int] = []
    stagnation_counter = 0
    stagnation_best: float | None = None

    rows: list[dict] = []

    for gen in sorted(gen_groups.keys()):
        gen_idx = gen_groups[gen]

        # --- Taboo check (before "evaluation") ---
        for idx in gen_idx:
            rows.append(
                {
                    "generation": gen,
                    "fitness": fitnesses[idx],
                    "is_taboo": _is_taboo(idx, taboo_indices, D, threshold),
                }
            )

        # --- Update taboo list (after "evaluation") ---
        gen_valid_f = [
            fitnesses[j]
            for j in gen_idx
            if not math.isnan(fitnesses[j]) and np.isfinite(fitnesses[j])
        ]
        all_seen_f = [
            fitnesses[j]
            for j in range(max(gen_idx) + 1)
            if not math.isnan(fitnesses[j]) and np.isfinite(fitnesses[j])
        ]

        if strategy == "always":
            taboo_indices.extend(gen_idx)

        elif strategy == "poor_fitness":
            if len(all_seen_f) >= 2:
                cutoff = (
                    np.percentile(all_seen_f, 100.0 - poor_pct)
                    if minimization
                    else np.percentile(all_seen_f, poor_pct)
                )
                for idx in gen_idx:
                    f = fitnesses[idx]
                    if math.isnan(f) or not np.isfinite(f):
                        continue
                    if (minimization and f >= cutoff) or (not minimization and f <= cutoff):
                        taboo_indices.append(idx)

        elif strategy == "stagnation":
            gen_best = (
                (min if minimization else max)(gen_valid_f) if gen_valid_f else None
            )
            if stagnation_best is None:
                stagnation_best = gen_best  # first generation: set baseline
            elif gen_best is not None:
                improved = (
                    gen_best < stagnation_best if minimization else gen_best > stagnation_best
                )
                stagnation_best = (
                    min(stagnation_best, gen_best)
                    if minimization
                    else max(stagnation_best, gen_best)
                )
                stagnation_counter = 0 if improved else stagnation_counter + 1
                if stagnation_counter >= stagnation_window:
                    taboo_indices.extend(gen_idx)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(sim_df: pd.DataFrame, config: dict, run_name: str) -> dict:
    valid = sim_df.dropna(subset=["fitness"])
    valid = valid[np.isfinite(valid["fitness"])]
    if valid.empty:
        return {}

    median_f = valid["fitness"].median()
    pruned = valid[valid["is_taboo"]]
    kept = valid[~valid["is_taboo"]]

    pruning_rate = len(pruned) / len(valid)

    # Precision: of pruned solutions, what fraction was below-median (bad)?
    if len(pruned) > 0:
        precision = (pruned["fitness"] < median_f).mean()
    else:
        precision = float("nan")

    # Recall: of below-median solutions, what fraction was pruned?
    n_bad = (valid["fitness"] < median_f).sum()
    if n_bad > 0:
        recall = (pruned["fitness"] < median_f).sum() / n_bad
    else:
        recall = float("nan")

    return {
        "run": run_name,
        "config": config_label(config),
        "strategy": config["strategy"],
        "threshold": config["threshold"],
        "stagnation_window": config.get("stagnation_window", None),
        "poor_fitness_percentile": config.get("poor_fitness_percentile", None),
        "n_total": len(valid),
        "n_pruned": len(pruned),
        "pruning_rate": round(pruning_rate, 4),
        "precision": round(precision, 4) if not math.isnan(precision) else float("nan"),
        "recall": round(recall, 4) if not math.isnan(recall) else float("nan"),
        "mean_fitness_pruned": round(float(pruned["fitness"].mean()), 6)
        if len(pruned) > 0
        else float("nan"),
        "mean_fitness_kept": round(float(kept["fitness"].mean()), 6)
        if len(kept) > 0
        else float("nan"),
        "best_fitness_kept": round(float(kept["fitness"].max()), 6)
        if len(kept) > 0
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing .jsonl run files (or a single .jsonl file).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("taboo_simulation_results.csv"),
        help="Output CSV path (default: taboo_simulation_results.csv).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.3],
        metavar="T",
        help="Similarity thresholds to sweep (default: 0.05 0.1 0.2 0.3).",
    )
    parser.add_argument(
        "--stagnation-windows",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        metavar="W",
        help="Stagnation window sizes (default: 3 5 10).",
    )
    parser.add_argument(
        "--poor-fitness-percentiles",
        nargs="+",
        type=float,
        default=[10.0, 25.0, 50.0],
        metavar="P",
        help="Poor-fitness percentile cut-offs (default: 10 25 50).",
    )
    parser.add_argument(
        "--minimization",
        action="store_true",
        help="Treat fitness as a minimization objective (default: maximization).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress.",
    )
    args = parser.parse_args()

    data_path = args.data_dir
    if data_path.is_file():
        jsonl_files = [data_path]
    else:
        jsonl_files = sorted(data_path.glob("*.jsonl"))

    if not jsonl_files:
        sys.exit(f"No .jsonl files found in {data_path}")

    configs = build_config_grid(
        thresholds=args.thresholds,
        stagnation_windows=args.stagnation_windows,
        poor_fitness_percentiles=args.poor_fitness_percentiles,
    )
    print(
        f"Found {len(jsonl_files)} run file(s), "
        f"{len(configs)} configurations → "
        f"{len(jsonl_files) * len(configs)} simulations total."
    )

    all_rows: list[dict] = []

    for fpath in jsonl_files:
        run_name = fpath.stem
        if args.verbose:
            print(f"\n--- {run_name} ---")

        records = load_run(fpath)
        if len(records) < 2:
            print(f"  Skipping {run_name}: fewer than 2 solutions.")
            continue

        if args.verbose:
            print(f"  {len(records)} solutions, computing distance matrix …")

        D = build_distance_matrix(records, verbose=args.verbose)

        for cfg in configs:
            sim_df = simulate_run(records, D, cfg, minimization=args.minimization)
            row = compute_metrics(sim_df, cfg, run_name)
            if row:
                all_rows.append(row)

        if not args.verbose:
            print(f"  {run_name}: done ({len(records)} solutions).")

    if not all_rows:
        sys.exit("No results produced — check your data files.")

    results = pd.DataFrame(all_rows)
    results.to_csv(args.out, index=False)
    print(f"\nFull results saved to {args.out}")

    # --- Summary: aggregate over runs, show mean metrics per config ---
    summary_cols = ["strategy", "threshold", "stagnation_window", "poor_fitness_percentile"]
    agg = (
        results.groupby(summary_cols, dropna=False)[
            ["pruning_rate", "precision", "recall", "mean_fitness_pruned", "mean_fitness_kept"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    agg = agg.sort_values(["strategy", "threshold"])

    print("\n=== Mean metrics across all runs ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
