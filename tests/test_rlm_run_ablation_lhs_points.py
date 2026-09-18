"""Coverage for run_ablation_lhs_points.py's harness logic -- CLI parsing,
the n_lhs_points x seed x holdout_fids matrix, and reuse of
run_ablation.py's by-variant mean/std rollup -- kept free of ``ioh``/GPU/
real-data dependencies by mocking ``run_one_lhs_points_variant`` entirely
(the actual train+eval pipeline is exercised only via real runs, per the
module's own docstring)."""

from unittest.mock import patch

from llamea.rlm_surrogate.run_ablation import DEFAULT_HOLDOUT_SETS, DEFAULT_SEEDS
from llamea.rlm_surrogate.run_ablation_lhs_points import (
    DEFAULT_LHS_POINTS,
    _build_arg_parser,
    run_lhs_points_ablation,
)


def test_default_lhs_points_has_three_variants_above_the_shipped_default():
    assert len(DEFAULT_LHS_POINTS) == 3
    assert DEFAULT_LHS_POINTS == sorted(DEFAULT_LHS_POINTS)
    assert min(DEFAULT_LHS_POINTS) > 20  # current shipped default


def test_cli_defaults_reuse_run_ablation_seeds_and_holdout_sets():
    args = _build_arg_parser().parse_args(
        ["--data-dir", "/data", "--output-dir", "/out"]
    )
    assert args.lhs_points_variants == DEFAULT_LHS_POINTS
    assert args.seeds == DEFAULT_SEEDS
    assert args.holdout_sets is None  # resolved to DEFAULT_HOLDOUT_SETS in main()
    assert args.feature_mode == "lhs"


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


def test_cli_lhs_points_override():
    args = _build_arg_parser().parse_args(
        [
            "--data-dir",
            "/data",
            "--output-dir",
            "/out",
            "--lhs-points",
            "10",
            "30",
        ]
    )
    assert args.lhs_points_variants == [10, 30]


def test_run_lhs_points_ablation_covers_full_matrix(tmp_path):
    """`run_lhs_points_ablation` must call `run_one_lhs_points_variant` once
    per (n_lhs_points, seed, holdout_fids) combination."""
    calls = []

    def fake_run_one(*, n_lhs_points, seed, holdout_fids, **kwargs):
        calls.append((n_lhs_points, seed, tuple(holdout_fids)))
        return {
            "variant": str(n_lhs_points),
            "n_lhs_points": n_lhs_points,
            "seed": seed,
            "holdout_fids": list(holdout_fids),
            "n_test": 10,
            "spearman_rho": 0.5,
            "kendall_tau": 0.3,
            "wall_clock_seconds": {"pipeline": 0.0, "train": 0.0, "eval": 0.0},
        }

    with patch(
        "llamea.rlm_surrogate.run_ablation_lhs_points.run_one_lhs_points_variant",
        side_effect=fake_run_one,
    ):
        results = run_lhs_points_ablation(
            data_dir="/data",
            output_dir=tmp_path,
            lhs_points_variants=[5, 20],
            seeds=[0, 1],
            holdout_sets=[[21, 22], [3, 8]],
        )

    assert len(calls) == 2 * 2 * 2  # lhs_points x seeds x holdout_sets
    assert set(calls) == {
        (n_points, seed, holdout)
        for n_points in (5, 20)
        for seed in (0, 1)
        for holdout in ((21, 22), (3, 8))
    }
    assert len(results) == 8

    summary_path = tmp_path / "ablation_summary.json"
    assert summary_path.exists()
