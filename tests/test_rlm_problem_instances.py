import pytest

from llamea.rlm_surrogate.problem_instances import (
    BBOB_DEFAULT,
    MA_BBOB_DEFAULT,
    InstanceSweepConfig,
    match_sweep_config,
)


def test_bbob_default_expected_length_matches_reference_script():
    # examples/black-box-optimization.py: dim=[5] x fid=1..24 x iid=[1,2,3] x rep=3
    assert BBOB_DEFAULT.expected_length == 1 * 24 * 3 * 3


def test_ma_bbob_default_expected_length_matches_reference_script():
    # benchmarks/ma_bbob/run_mabbob.py: dim=[2,5] x idx=0..99
    assert MA_BBOB_DEFAULT.expected_length == 2 * 100


def test_decode_first_and_last_index_bbob():
    first = BBOB_DEFAULT.decode(0)
    assert first == {"dim": 5, "fid_or_idx": 1, "iid": 1, "rep": 0}
    last = BBOB_DEFAULT.decode(BBOB_DEFAULT.expected_length - 1)
    assert last == {"dim": 5, "fid_or_idx": 24, "iid": 3, "rep": 2}


def test_decode_covers_dim_boundary_ma_bbob():
    # First 100 indices are dim=2, next 100 are dim=5.
    assert MA_BBOB_DEFAULT.decode(0)["dim"] == 2
    assert MA_BBOB_DEFAULT.decode(99)["dim"] == 2
    assert MA_BBOB_DEFAULT.decode(100)["dim"] == 5
    assert MA_BBOB_DEFAULT.decode(199)["dim"] == 5


def test_decode_out_of_range_raises():
    with pytest.raises(IndexError):
        BBOB_DEFAULT.decode(BBOB_DEFAULT.expected_length)
    with pytest.raises(IndexError):
        BBOB_DEFAULT.decode(-1)


def test_decode_is_bijective_small_config():
    cfg = InstanceSweepConfig(
        kind="bbob", dims=[2, 3], fids_or_idxs=[1, 2, 3], iids=[1, 2]
    )
    assert cfg.expected_length == 2 * 3 * 2 * 1
    decoded = [cfg.decode(i) for i in range(cfg.expected_length)]
    as_tuples = {tuple(sorted(d.items())) for d in decoded}
    assert len(as_tuples) == cfg.expected_length  # every combo appears exactly once


def test_match_sweep_config_picks_length_match():
    cfg = match_sweep_config(
        BBOB_DEFAULT.expected_length, [BBOB_DEFAULT, MA_BBOB_DEFAULT]
    )
    assert cfg is BBOB_DEFAULT
    cfg2 = match_sweep_config(
        MA_BBOB_DEFAULT.expected_length, [BBOB_DEFAULT, MA_BBOB_DEFAULT]
    )
    assert cfg2 is MA_BBOB_DEFAULT


def test_match_sweep_config_returns_none_on_no_match():
    assert match_sweep_config(7, [BBOB_DEFAULT, MA_BBOB_DEFAULT]) is None


def test_instance_sweep_config_yaml_roundtrip(tmp_path):
    cfg = InstanceSweepConfig(
        kind="ma_bbob", dims=[2], fids_or_idxs=[0, 1], iids=[1], reps=1
    )
    path = tmp_path / "sweep.yaml"
    cfg.to_yaml(path)
    loaded = InstanceSweepConfig.from_yaml(path)
    assert loaded == cfg


# --- Tests that need the real `ioh` extra (uv sync --group rlm-surrogate) ---

ioh = pytest.importorskip("ioh")


def test_reconstruct_bbob_problem_and_fingerprint():
    from llamea.rlm_surrogate.problem_instances import (
        compute_problem_feature_text,
        reconstruct_problem,
    )

    cfg = InstanceSweepConfig(kind="bbob", dims=[2], fids_or_idxs=[1], iids=[1], reps=1)
    decoded = cfg.decode(0)
    problem = reconstruct_problem(cfg, decoded)
    assert list(problem.bounds.lb) == [-5.0, -5.0]

    text = compute_problem_feature_text(cfg, 0, n_points=3, seed=0)
    assert text.startswith("LHS[3pts,dim=2]:")
    assert text.count("->") == 3


def test_reconstruct_ma_bbob_problem_and_fingerprint():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    cfg = InstanceSweepConfig(
        kind="ma_bbob", dims=[2], fids_or_idxs=[0, 1], iids=[1], reps=1
    )
    text = compute_problem_feature_text(cfg, 1, n_points=2, seed=0)
    assert text.startswith("LHS[2pts,dim=2]:")


def test_fingerprint_deterministic_for_same_seed():
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    cfg = InstanceSweepConfig(kind="bbob", dims=[2], fids_or_idxs=[3], iids=[1], reps=1)
    a = compute_problem_feature_text(cfg, 0, n_points=5, seed=42)
    b = compute_problem_feature_text(cfg, 0, n_points=5, seed=42)
    assert a == b


def test_fingerprint_shares_probe_locations_across_instances_same_dim():
    """Same seed + dim -> same x-coordinates probed, per lhs_fingerprint's
    docstring guarantee (differences reflect the function, not sampling)."""
    from llamea.rlm_surrogate.problem_instances import compute_problem_feature_text

    cfg = InstanceSweepConfig(
        kind="bbob", dims=[2], fids_or_idxs=[1, 2], iids=[1], reps=1
    )
    text_f1 = compute_problem_feature_text(cfg, 0, n_points=3, seed=7)
    text_f2 = compute_problem_feature_text(cfg, 1, n_points=3, seed=7)
    coords_f1 = [row.split(")->")[0] for row in text_f1.split(": ", 1)[1].split("; ")]
    coords_f2 = [row.split(")->")[0] for row in text_f2.split(": ", 1)[1].split("; ")]
    assert coords_f1 == coords_f2
