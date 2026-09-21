"""Coverage for run_ablation_lhs_points.py's harness logic -- CLI parsing,
the n_lhs_points x seed x holdout_fids matrix, reuse of run_ablation.py's
by-variant mean/std rollup, and restartability (resume-by-default via
cached ablation_result.json files, reusing run_ablation.py's helpers) --
kept free of ``ioh``/GPU/real-data dependencies by mocking
``run_one_lhs_points_variant`` entirely (the actual train+eval pipeline is
exercised only via real runs, per the module's own docstring)."""

from unittest.mock import patch

from llamea.rlm_surrogate.run_ablation import (
    DEFAULT_HOLDOUT_SETS,
    DEFAULT_SEEDS,
    _run_dir_for,
    _write_json_atomic,
)
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


def test_cli_force_rerun_defaults_to_false():
    args = _build_arg_parser().parse_args(
        ["--data-dir", "/data", "--output-dir", "/out"]
    )
    assert args.force_rerun is False


def test_run_lhs_points_ablation_resume_skips_cached_combinations(tmp_path):
    """Mirrors run_ablation.py's resume test: a pre-existing, complete
    ablation_result.json for one (n_lhs_points, seed, holdout_fids)
    combination is reused instead of retraining."""
    cached_run_dir = _run_dir_for(tmp_path, "lhs50", 0, [21, 22])
    cached_run_dir.mkdir(parents=True)
    cached_result = {
        "variant": "50",
        "n_lhs_points": 50,
        "seed": 0,
        "holdout_fids": [21, 22],
        "spearman_rho": 0.999,
        "kendall_tau": 0.888,
        "n_test": 42,
        "wall_clock_seconds": {"pipeline": 0.0, "train": 0.0, "eval": 0.0},
    }
    _write_json_atomic(cached_run_dir / "ablation_result.json", cached_result)

    calls = []

    def fake_run_one(*, n_lhs_points, seed, holdout_fids, **kwargs):
        calls.append((n_lhs_points, seed, tuple(holdout_fids)))
        return {
            "variant": str(n_lhs_points),
            "n_lhs_points": n_lhs_points,
            "seed": seed,
            "holdout_fids": list(holdout_fids),
            "n_test": 10,
            "spearman_rho": 0.1,
            "kendall_tau": 0.1,
            "wall_clock_seconds": {"pipeline": 0.0, "train": 0.0, "eval": 0.0},
        }

    with patch(
        "llamea.rlm_surrogate.run_ablation_lhs_points.run_one_lhs_points_variant",
        side_effect=fake_run_one,
    ):
        results = run_lhs_points_ablation(
            data_dir="/data",
            output_dir=tmp_path,
            lhs_points_variants=[50],
            seeds=[0],
            holdout_sets=[[21, 22], [3, 8]],
        )

    assert calls == [(50, 0, (3, 8))]
    assert cached_result in results
    assert len(results) == 2


def test_run_lhs_points_ablation_force_rerun_ignores_cached_results(tmp_path):
    cached_run_dir = _run_dir_for(tmp_path, "lhs50", 0, [21, 22])
    cached_run_dir.mkdir(parents=True)
    _write_json_atomic(
        cached_run_dir / "ablation_result.json",
        {
            "variant": "50",
            "n_lhs_points": 50,
            "seed": 0,
            "holdout_fids": [21, 22],
            "spearman_rho": 0.999,
            "kendall_tau": 0.888,
            "n_test": 42,
            "wall_clock_seconds": {"pipeline": 0.0, "train": 0.0, "eval": 0.0},
        },
    )

    calls = []

    def fake_run_one(*, n_lhs_points, seed, holdout_fids, **kwargs):
        calls.append((n_lhs_points, seed, tuple(holdout_fids)))
        return {
            "variant": str(n_lhs_points),
            "n_lhs_points": n_lhs_points,
            "seed": seed,
            "holdout_fids": list(holdout_fids),
            "n_test": 10,
            "spearman_rho": 0.1,
            "kendall_tau": 0.1,
            "wall_clock_seconds": {"pipeline": 0.0, "train": 0.0, "eval": 0.0},
        }

    with patch(
        "llamea.rlm_surrogate.run_ablation_lhs_points.run_one_lhs_points_variant",
        side_effect=fake_run_one,
    ):
        results = run_lhs_points_ablation(
            data_dir="/data",
            output_dir=tmp_path,
            lhs_points_variants=[50],
            seeds=[0],
            holdout_sets=[[21, 22]],
            resume=False,
        )

    assert calls == [(50, 0, (21, 22))]
    assert results[0]["spearman_rho"] == 0.1
