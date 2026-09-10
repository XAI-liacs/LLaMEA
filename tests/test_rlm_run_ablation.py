"""Coverage for run_ablation.py's harness logic -- CLI parsing, the
variant x seed x holdout_fids matrix, and the by-variant mean/std rollup --
kept free of ``ioh``/GPU/real-data dependencies by mocking ``run_one_variant``
entirely (the actual train+eval pipeline is exercised only via real runs, per
the module's own docstring)."""

from unittest.mock import patch

from llamea.rlm_surrogate.run_ablation import (
    DEFAULT_HOLDOUT_SETS,
    DEFAULT_SEEDS,
    _build_arg_parser,
    _summarize_by_variant,
    run_ablation,
)


def test_default_holdout_sets_and_seeds_have_more_than_one_entry():
    """The whole point of this change: rank variants across several seeds
    and several held-out regions, not trust a single run."""
    assert len(DEFAULT_SEEDS) >= 5
    assert len(DEFAULT_HOLDOUT_SETS) >= 3
    assert len({tuple(h) for h in DEFAULT_HOLDOUT_SETS}) == len(DEFAULT_HOLDOUT_SETS)


def test_cli_holdout_fids_defaults_to_none_until_resolved_in_main():
    args = _build_arg_parser().parse_args(
        ["--data-dir", "/data", "--output-dir", "/out"]
    )
    assert args.holdout_sets is None
    assert args.seeds == DEFAULT_SEEDS


def test_cli_holdout_fids_repeatable_flag_builds_multiple_sets():
    args = _build_arg_parser().parse_args(
        [
            "--data-dir",
            "/data",
            "--output-dir",
            "/out",
            "--holdout-fids",
            "21",
            "22",
            "--holdout-fids",
            "3",
            "8",
        ]
    )
    assert args.holdout_sets == [[21, 22], [3, 8]]


def test_summarize_by_variant_mean_and_std():
    results = [
        {"variant": "lhs", "spearman_rho": 0.5, "kendall_tau": 0.3},
        {"variant": "lhs", "spearman_rho": 0.7, "kendall_tau": 0.5},
        {"variant": "meta+lhs", "spearman_rho": 0.4, "kendall_tau": 0.2},
    ]
    summary = _summarize_by_variant(results)
    assert summary["lhs"]["n_runs"] == 2
    assert summary["lhs"]["spearman_mean"] == 0.6
    assert summary["lhs"]["spearman_std"] > 0
    assert summary["meta+lhs"]["n_runs"] == 1
    assert summary["meta+lhs"]["spearman_std"] == 0.0  # single run -> no spread


def test_run_ablation_covers_full_variant_seed_holdout_matrix(tmp_path):
    """`run_ablation` must call `run_one_variant` once per (variant, seed,
    holdout_fids) combination -- the entire point of the requested change
    was benchmarking every combination, not just every (variant, seed)."""
    calls = []

    def fake_run_one_variant(*, variant, seed, holdout_fids, **kwargs):
        calls.append((variant, seed, tuple(holdout_fids)))
        return {
            "variant": variant,
            "seed": seed,
            "holdout_fids": list(holdout_fids),
            "n_test": 10,
            "spearman_rho": 0.5,
            "kendall_tau": 0.3,
            "wall_clock_seconds": {"pipeline": 0.0, "train": 0.0, "eval": 0.0},
        }

    with patch(
        "llamea.rlm_surrogate.run_ablation.run_one_variant",
        side_effect=fake_run_one_variant,
    ):
        results = run_ablation(
            data_dir="/data",
            output_dir=tmp_path,
            variants=["lhs", "meta+lhs"],
            seeds=[0, 1],
            holdout_sets=[[21, 22], [3, 8]],
        )

    assert len(calls) == 2 * 2 * 2  # variants x seeds x holdout_sets
    assert set(calls) == {
        (variant, seed, holdout)
        for variant in ("lhs", "meta+lhs")
        for seed in (0, 1)
        for holdout in ((21, 22), (3, 8))
    }
    assert len(results) == 8

    summary_path = tmp_path / "ablation_summary.json"
    assert summary_path.exists()
