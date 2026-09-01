import json
from types import SimpleNamespace

import pytest

from llamea.rlm_surrogate.problem_instances import (
    ProblemInstance,
    _parse_range_string,
    instances_from_performance_data,
    load_experiment_instances,
    parse_training_instances,
    resolve_instances_for_record,
)


def _record(metadata, source_file):
    return SimpleNamespace(metadata=metadata, source_file=source_file)


def test_instances_from_performance_data():
    performance_data = [
        {"fid": 1, "iid": 2, "dim": 10, "auc": 0.8},
        {"fid": 3, "iid": 1, "dim": 10, "auc": 0.5},
    ]
    instances = instances_from_performance_data(performance_data)
    assert instances == [
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=1, iid=2),
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=3, iid=1),
    ]


def test_parse_range_string_valid_and_invalid():
    assert _parse_range_string("range(0, 10)") == list(range(0, 10))
    assert _parse_range_string(" range(2, 5) ") == [2, 3, 4]
    assert _parse_range_string("not a range") is None
    assert _parse_range_string("range(0, 10, 2)") is None


def test_parse_training_instances_bbob_pairs():
    spec = {
        "name": "BBOB",
        "dims": [10],
        "training_instances": [[1, 1], [1, 2], [2, 1]],
    }
    assert parse_training_instances(spec) == [
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=1, iid=1),
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=1, iid=2),
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=2, iid=1),
    ]


def test_parse_training_instances_ma_bbob_range_string():
    spec = {"name": "MA_BBOB", "dims": [10], "training_instances": "range(0, 10)"}
    instances = parse_training_instances(spec)
    assert instances == [
        ProblemInstance(kind="ma_bbob", dim=10, fid_or_idx=i) for i in range(10)
    ]


def test_parse_training_instances_ma_bbob_list():
    spec = {"name": "MA_BBOB", "dims": [5], "training_instances": [0, 1, 2]}
    assert parse_training_instances(spec) == [
        ProblemInstance(kind="ma_bbob", dim=5, fid_or_idx=0),
        ProblemInstance(kind="ma_bbob", dim=5, fid_or_idx=1),
        ProblemInstance(kind="ma_bbob", dim=5, fid_or_idx=2),
    ]


def test_parse_training_instances_ambiguous_dims_returns_none():
    spec = {"name": "MA_BBOB", "dims": [2, 5], "training_instances": "range(0, 10)"}
    assert parse_training_instances(spec) is None
    assert (
        parse_training_instances({"name": "BBOB", "dims": [], "training_instances": []})
        is None
    )


def test_parse_training_instances_unknown_name_returns_none():
    spec = {"name": "SOMETHING_ELSE", "dims": [10], "training_instances": []}
    assert parse_training_instances(spec) is None


def test_parse_training_instances_malformed_bbob_pairs_returns_none():
    spec = {"name": "BBOB", "dims": [10], "training_instances": "not-a-list"}
    assert parse_training_instances(spec) is None


def test_load_experiment_instances(tmp_path):
    log_path = tmp_path / "experimentlog.jsonl"
    lines = [
        {
            "log_dir": "run-A-1",
            "problem": {
                "name": "BBOB",
                "dims": [10],
                "training_instances": [[1, 1], [1, 2]],
            },
        },
        {
            "log_dir": "run-B-2",
            "problem": {
                "name": "MA_BBOB",
                "dims": [10],
                "training_instances": "range(0, 3)",
            },
        },
        {"not_log_dir": "junk"},
        "not even json",
    ]
    with open(log_path, "w") as fh:
        for line in lines:
            fh.write((json.dumps(line) if not isinstance(line, str) else line) + "\n")

    result = load_experiment_instances(log_path)
    assert set(result) == {"run-A-1", "run-B-2"}
    assert result["run-A-1"] == [
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=1, iid=1),
        ProblemInstance(kind="bbob", dim=10, fid_or_idx=1, iid=2),
    ]
    assert result["run-B-2"] == [
        ProblemInstance(kind="ma_bbob", dim=10, fid_or_idx=i) for i in range(3)
    ]


def test_load_experiment_instances_missing_file_returns_empty(tmp_path):
    assert load_experiment_instances(tmp_path / "does_not_exist.jsonl") == {}


def test_resolve_instances_for_record_uses_performance_data_without_file_io():
    record = _record(
        metadata={
            "performance_data": [
                {"fid": 4, "iid": 1, "dim": 10, "auc": 0.9},
            ]
        },
        source_file="/nonexistent/experiment/run-1/log.jsonl",
    )
    result = resolve_instances_for_record(record, {})
    assert result == [ProblemInstance(kind="bbob", dim=10, fid_or_idx=4, iid=1)]


def test_resolve_instances_for_record_falls_back_to_experimentlog(tmp_path):
    experiment_dir = tmp_path / "MA_BBOB_experiment"
    run_dir = experiment_dir / "run-LLaMEA-MA_BBOB-20"
    run_dir.mkdir(parents=True)
    (run_dir / "log.jsonl").write_text("")
    experimentlog = experiment_dir / "experimentlog.jsonl"
    experimentlog.write_text(
        json.dumps(
            {
                "log_dir": "run-LLaMEA-MA_BBOB-20",
                "problem": {
                    "name": "MA_BBOB",
                    "dims": [10],
                    "training_instances": "range(0, 2)",
                },
            }
        )
        + "\n"
    )

    record = _record(metadata={}, source_file=str(run_dir / "log.jsonl"))
    cache = {}
    result = resolve_instances_for_record(record, cache)
    assert result == [
        ProblemInstance(kind="ma_bbob", dim=10, fid_or_idx=0),
        ProblemInstance(kind="ma_bbob", dim=10, fid_or_idx=1),
    ]
    # Second call for a record in the same experiment folder reuses the cache
    # (no re-read needed) -- verify by checking the cache was populated once.
    assert str(experiment_dir) in cache
    resolve_instances_for_record(record, cache)
    assert len(cache) == 1


def test_resolve_instances_for_record_returns_none_when_unresolvable(tmp_path):
    experiment_dir = tmp_path / "some_experiment"
    run_dir = experiment_dir / "run-1"
    run_dir.mkdir(parents=True)
    record = _record(metadata={}, source_file=str(run_dir / "log.jsonl"))
    assert resolve_instances_for_record(record, {}) is None


# --- Tier A: static meta-features. No `ioh` needed (pandas only, for the
# MA-BBOB weights.csv lookup). ---


def test_compute_meta_feature_text_bbob():
    from llamea.rlm_surrogate.problem_instances import compute_meta_feature_text

    instance = ProblemInstance(kind="bbob", dim=10, fid_or_idx=1, iid=1)
    text = compute_meta_feature_text(instance)
    assert "family: BBOB" in text
    assert "dim: 10" in text
    assert "Sphere" in text
    assert "separable=True" in text


def test_compute_meta_feature_text_ma_bbob_reads_real_weights_csv():
    from llamea.rlm_surrogate.problem_instances import compute_meta_feature_text

    instance = ProblemInstance(kind="ma_bbob", dim=5, fid_or_idx=0)
    text = compute_meta_feature_text(instance)
    assert "family: MA-BBOB" in text
    assert "dim: 5" in text
    assert "MA-BBOB combination:" in text
    assert "+" in text  # at least two weighted components, per row 0 of weights.csv


def test_describe_ma_bbob_composition_matches_real_weights_csv_row0():
    from llamea.rlm_surrogate.problem_instances import describe_ma_bbob_composition

    # benchmarks/ma_bbob/weights.csv row 0: f14 weight ~0.501, f23 weight
    # ~0.499, all other columns 0 -- verified by hand reading the CSV.
    text = describe_ma_bbob_composition(0)
    assert "f14" in text
    assert "f23" in text
    assert "50." in text  # both components round to ~50.0%/50.1%/49.9%


def test_compute_problem_feature_text_meta_mode_needs_no_ioh():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=1, iid=1)
    text = compute_problem_feature_text(instance, mode="meta")
    assert "family: BBOB" in text
    assert "LHS" not in text


def test_compute_problem_feature_text_unknown_mode_raises():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=1, iid=1)
    with pytest.raises(ValueError):
        compute_problem_feature_text(instance, mode="bogus")


# --- Tests that need the real `ioh` extra (uv sync --group rlm-surrogate) ---

ioh = pytest.importorskip("ioh")


def test_reconstruct_bbob_problem_and_fingerprint():
    from llamea.rlm_surrogate.problem_instances import (
        compute_problem_feature_text,
        reconstruct_problem,
    )

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=1, iid=1)
    problem = reconstruct_problem(instance)
    assert list(problem.bounds.lb) == [-5.0, -5.0]

    text = compute_problem_feature_text(instance, n_points=3, seed=0)
    assert text.startswith("LHS[3pts,dim=2]:")
    assert text.count("->") == 3


def test_reconstruct_ma_bbob_problem_and_fingerprint():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="ma_bbob", dim=2, fid_or_idx=1)
    text = compute_problem_feature_text(instance, n_points=2, seed=0)
    assert text.startswith("LHS[2pts,dim=2]:")


def test_fingerprint_deterministic_for_same_seed():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=3, iid=1)
    a = compute_problem_feature_text(instance, n_points=5, seed=42)
    b = compute_problem_feature_text(instance, n_points=5, seed=42)
    assert a == b


def test_fingerprint_shares_probe_locations_across_instances_same_dim():
    """Same seed + dim -> same x-coordinates probed, per lhs_fingerprint's
    docstring guarantee (differences reflect the function, not sampling)."""
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance_f1 = ProblemInstance(kind="bbob", dim=2, fid_or_idx=1, iid=1)
    instance_f2 = ProblemInstance(kind="bbob", dim=2, fid_or_idx=2, iid=1)
    text_f1 = compute_problem_feature_text(instance_f1, n_points=3, seed=7)
    text_f2 = compute_problem_feature_text(instance_f2, n_points=3, seed=7)
    coords_f1 = [row.split(")->")[0] for row in text_f1.split(": ", 1)[1].split("; ")]
    coords_f2 = [row.split(")->")[0] for row in text_f2.split(": ", 1)[1].split("; ")]
    assert coords_f1 == coords_f2


# --- Tier B: computed LHS summary stats (same sample, different rendering). ---


def test_lhs_summary_stats_shape_and_fields():
    from llamea.rlm_surrogate.problem_instances import (
        lhs_summary_stats,
        reconstruct_problem,
    )

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=1, iid=1)
    problem = reconstruct_problem(instance)
    text = lhs_summary_stats(problem, dim=2, n_points=20, seed=0)
    assert text.startswith("STATS[20pts,dim=2]:")
    for field in (
        "y_mean",
        "y_std",
        "y_skew",
        "y_kurtosis",
        "cv",
        "quad_r2",
        "cond_est",
    ):
        assert f"{field}=" in text


def test_lhs_summary_stats_deterministic_for_same_seed():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=3, iid=1)
    a = compute_problem_feature_text(instance, n_points=20, seed=42, mode="lhs_stats")
    b = compute_problem_feature_text(instance, n_points=20, seed=42, mode="lhs_stats")
    assert a == b


def test_compute_problem_feature_text_meta_plus_lhs_concatenates():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="bbob", dim=2, fid_or_idx=1, iid=1)
    text = compute_problem_feature_text(instance, n_points=3, seed=0, mode="meta+lhs")
    assert "family: BBOB" in text
    assert "LHS[3pts,dim=2]:" in text


def test_compute_problem_feature_text_meta_plus_lhs_stats_concatenates():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    instance = ProblemInstance(kind="ma_bbob", dim=2, fid_or_idx=1)
    text = compute_problem_feature_text(
        instance, n_points=20, seed=0, mode="meta+lhs_stats"
    )
    assert "family: MA-BBOB" in text
    assert "STATS[20pts,dim=2]:" in text


def test_lhs_summary_stats_quadratic_recovers_sphere_conditioning():
    """Sphere (f1) is an isotropic quadratic bowl -- the diagonal-quadratic
    meta-model should fit it almost perfectly (high R^2) with roughly equal
    per-axis coefficients (cond_est close to 1)."""
    from llamea.rlm_surrogate.problem_instances import (
        lhs_summary_stats,
        reconstruct_problem,
    )

    instance = ProblemInstance(kind="bbob", dim=3, fid_or_idx=1, iid=1)
    problem = reconstruct_problem(instance)
    text = lhs_summary_stats(problem, dim=3, n_points=30, seed=0)
    fields = dict(part.strip().split("=") for part in text.split(": ", 1)[1].split(";"))
    assert float(fields["quad_r2"]) > 0.99
    assert float(fields["cond_est"]) < 2.0
