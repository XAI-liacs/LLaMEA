import json
from pathlib import Path

import pytest

from llamea.rlm_surrogate.data_pipeline import (
    RLMExample,
    SplitConfig,
    build_x,
    build_y,
    explode_aucs_with_problem_features,
    filter_errored,
    leave_function_out_split,
    lineage_generation_split,
    make_examples,
    read_examples_jsonl,
    run_pipeline,
    write_examples_jsonl,
)
from llamea.rlm_surrogate.schema import BladeRecord, iter_blade_records, load_directory

FIXTURES = Path(__file__).parent / "fixtures" / "rlm"


@pytest.fixture()
def records():
    return load_directory(FIXTURES)


def test_filter_errored_drops_only_errored(records):
    kept, n_dropped = filter_errored(records)
    assert n_dropped == sum(1 for r in records if r.has_error)
    assert all(not r.has_error for r in kept)
    assert len(kept) + n_dropped == len(records)


def test_build_x_default_includes_description_and_code():
    r = next(iter_blade_records(FIXTURES / "run_alpha.jsonl"))
    x = build_x(r)
    assert r.description in x
    assert r.code in x
    assert "# Code" in x


def test_build_x_code_only_ablation():
    r = next(iter_blade_records(FIXTURES / "run_alpha.jsonl"))
    x = build_x(r, include_description=False, include_configspace=False)
    assert r.description not in x
    assert r.code in x


def test_build_x_never_leaks_feedback_or_fitness():
    r = next(iter_blade_records(FIXTURES / "run_alpha.jsonl"))
    x = build_x(r)
    assert r.feedback not in x


def test_build_y_fitness_and_aucs():
    r = next(iter_blade_records(FIXTURES / "run_beta.jsonl"))
    assert build_y(r, target="fitness") == r.fitness
    assert build_y(r, target="aucs") == r.aucs


def test_make_examples_skips_missing_aucs_but_keeps_fitness():
    records = list(iter_blade_records(FIXTURES / "run_alpha.jsonl"))
    kept, _ = filter_errored(records)
    fitness_examples = make_examples(kept, target="fitness")
    aucs_examples = make_examples(kept, target="aucs")
    assert len(fitness_examples) == len(kept)
    assert len(aucs_examples) == len(
        kept
    )  # fixture always populates aucs when not errored
    assert all(isinstance(e.y, float) for e in fitness_examples)
    assert all(isinstance(e.y, list) for e in aucs_examples)


def test_lineage_generation_split_whole_run_holdout(records):
    kept, _ = filter_errored(records)
    examples = make_examples(kept)
    result = lineage_generation_split(examples, SplitConfig(seed=0))
    assert result.log["strategy"] == "whole_run_holdout_for_test"
    test_runs = set(result.log["test_runs"])
    assert test_runs
    # Every test example must come from a held-out run.
    assert all(e.run_id in test_runs for e in result.test)
    # No run in the held-out test set contributes to train/val.
    train_val_runs = {e.run_id for e in result.train} | {e.run_id for e in result.val}
    assert train_val_runs.isdisjoint(test_runs)


def test_lineage_generation_split_prefers_later_generations_for_val(records):
    kept, _ = filter_errored(records)
    examples = make_examples(kept)
    result = lineage_generation_split(examples, SplitConfig(seed=0))
    val_by_run = {}
    train_by_run = {}
    for e in result.val:
        val_by_run.setdefault(e.run_id, []).append(e.generation)
    for e in result.train:
        train_by_run.setdefault(e.run_id, []).append(e.generation)
    for run_id, val_gens in val_by_run.items():
        if run_id not in train_by_run:
            continue
        # Held-out generations should be >= the max training generation for
        # that run (later generations preferred over random interspersion).
        assert min(val_gens) >= max(train_by_run[run_id]) - 0


def test_lineage_generation_split_within_run_when_too_few_runs():
    kept, _ = filter_errored(list(iter_blade_records(FIXTURES / "run_alpha.jsonl")))
    examples = make_examples(kept)
    result = lineage_generation_split(
        examples, SplitConfig(min_runs_for_file_holdout=3, seed=0)
    )
    assert result.log["strategy"] == "within_run_generation_holdout"
    assert result.test
    assert result.train


def test_split_never_loses_or_duplicates_examples(records):
    kept, _ = filter_errored(records)
    examples = make_examples(kept)
    result = lineage_generation_split(examples, SplitConfig(seed=0))
    all_ids = [e.id for e in result.train + result.val + result.test]
    assert len(all_ids) == len(examples)
    assert len(set(all_ids)) == len(all_ids)


def test_write_and_read_examples_roundtrip(tmp_path):
    examples = [
        RLMExample(
            id="a",
            run_id="r",
            generation=0,
            parent_ids=[],
            x="code",
            y=0.5,
            fitness=0.5,
        )
    ]
    path = tmp_path / "out.jsonl"
    write_examples_jsonl(examples, path)
    loaded = read_examples_jsonl(path)
    assert loaded == examples


def test_run_pipeline_end_to_end(tmp_path):
    summary = run_pipeline(FIXTURES, tmp_path)
    assert summary["n_files"] == 3
    assert summary["n_examples_total"] > 0
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "stats.json").exists()

    stats_on_disk = json.loads((tmp_path / "stats.json").read_text())
    assert stats_on_disk["n_examples_total"] == summary["n_examples_total"]
    assert "warnings" in stats_on_disk  # small-sample-size flag should fire

    train = read_examples_jsonl(tmp_path / "train.jsonl")
    test = read_examples_jsonl(tmp_path / "test.jsonl")
    assert set(e.run_id for e in train).isdisjoint(
        r for r in summary["split"].get("test_runs", [])
    )
    assert (
        len(train) + len(read_examples_jsonl(tmp_path / "val.jsonl")) + len(test)
        == summary["n_examples_total"]
    )


def test_run_pipeline_raises_on_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        run_pipeline(empty_dir, tmp_path / "out")


# --- explode_aucs_with_problem_features (needs the `ioh` extra) ---

ioh = pytest.importorskip("ioh")

_PERF_DATA_2D = [
    {"fid": 1, "iid": 1, "dim": 2, "auc": 0.0},
    {"fid": 1, "iid": 2, "dim": 2, "auc": 0.0},
]


def _blade_record(
    id_,
    aucs,
    *,
    metadata_extra=None,
    run_id="r1",
    generation=0,
    parent_ids=None,
    source_file="x.jsonl",
):
    metadata = dict(metadata_extra or {})
    if aucs is not None:
        metadata["aucs"] = aucs
    return BladeRecord(
        id=id_,
        fitness=0.5,
        name="Algo",
        description="desc",
        code="class Algo:\n    pass\n",
        configspace="",
        generation=generation,
        feedback="",
        error="",
        parent_ids=parent_ids or [],
        operator=None,
        metadata=metadata,
        run_id=run_id,
        problem_id="BBOB",
        source_file=source_file,
        line_no=1,
    )


def test_explode_aucs_with_problem_features_cardinality_and_fields():
    records = [
        _blade_record(
            "cand1", [0.1, 0.2], metadata_extra={"performance_data": _PERF_DATA_2D}
        ),
        _blade_record(
            "cand2", [0.3, 0.4], metadata_extra={"performance_data": _PERF_DATA_2D}
        ),
    ]
    examples, counts = explode_aucs_with_problem_features(
        records, n_lhs_points=2, lhs_seed=0
    )
    assert counts == {
        "n_no_aucs": 0,
        "n_no_instance_mapping": 0,
        "n_length_mismatch": 0,
        "n_instance_errors": 0,
        "n_exploded": 4,
    }
    assert len(examples) == 4
    assert {e.id for e in examples} == {"cand1#0", "cand1#1", "cand2#0", "cand2#1"}
    for e in examples:
        assert e.candidate_id in ("cand1", "cand2")
        assert e.instance_index in (0, 1)
        assert "# Problem" in e.x
        assert "LHS[2pts" in e.x
    cand1_ys = sorted(e.y for e in examples if e.candidate_id == "cand1")
    assert cand1_ys == [0.1, 0.2]


def test_explode_aucs_with_problem_features_skips_missing_and_unmatched():
    records = [
        _blade_record("no_aucs", None),
        _blade_record(
            "no_mapping", [0.1, 0.2]
        ),  # no performance_data, no experimentlog
        _blade_record(
            "wrong_length",
            [0.1, 0.2, 0.3],  # length 3, performance_data has 2 entries
            metadata_extra={"performance_data": _PERF_DATA_2D},
        ),
        _blade_record(
            "ok", [0.5, 0.6], metadata_extra={"performance_data": _PERF_DATA_2D}
        ),
    ]
    examples, counts = explode_aucs_with_problem_features(records, n_lhs_points=2)
    assert counts["n_no_aucs"] == 1
    assert counts["n_no_instance_mapping"] == 1
    assert counts["n_length_mismatch"] == 1
    assert counts["n_exploded"] == 2
    assert {e.candidate_id for e in examples} == {"ok"}


def test_explode_aucs_with_problem_features_falls_back_to_experimentlog(tmp_path):
    experiment_dir = tmp_path / "MA_BBOB_experiment"
    run_dir = experiment_dir / "run-LLaMEA-MA_BBOB-1"
    run_dir.mkdir(parents=True)
    (experiment_dir / "experimentlog.jsonl").write_text(
        json.dumps(
            {
                "log_dir": "run-LLaMEA-MA_BBOB-1",
                "problem": {
                    "name": "MA_BBOB",
                    "dims": [2],
                    "training_instances": "range(0, 2)",
                },
            }
        )
        + "\n"
    )
    records = [
        _blade_record("cand1", [0.1, 0.2], source_file=str(run_dir / "log.jsonl"))
    ]
    examples, counts = explode_aucs_with_problem_features(records, n_lhs_points=2)
    assert counts["n_no_instance_mapping"] == 0
    assert counts["n_exploded"] == 2
    assert {e.id for e in examples} == {"cand1#0", "cand1#1"}


def test_explode_aucs_with_problem_features_preserves_lineage_fields():
    records = [
        _blade_record(
            "cand1",
            [0.1, 0.2],
            metadata_extra={"performance_data": _PERF_DATA_2D},
            run_id="r9",
            generation=3,
            parent_ids=["p1"],
        )
    ]
    examples, _ = explode_aucs_with_problem_features(records)
    assert all(e.run_id == "r9" for e in examples)
    assert all(e.generation == 3 for e in examples)
    assert all(e.parent_ids == ["p1"] for e in examples)
    assert all(
        e.fitness == 0.5 for e in examples
    )  # candidate-level fitness carried through


def test_explode_aucs_with_problem_features_meta_mode_sets_instance_fields():
    records = [
        _blade_record(
            "cand1", [0.1, 0.2], metadata_extra={"performance_data": _PERF_DATA_2D}
        )
    ]
    examples, counts = explode_aucs_with_problem_features(records, feature_mode="meta")
    assert counts["n_exploded"] == 2
    assert all(e.instance_kind == "bbob" for e in examples)
    assert {e.instance_fid_or_idx for e in examples} == {1}  # both entries are fid=1
    assert all("family: BBOB" in e.x for e in examples)
    assert all("LHS" not in e.x for e in examples)


# --- leave_function_out_split ---


def _exploded_example(id_, instance_kind, instance_fid_or_idx, generation=0):
    return RLMExample(
        id=id_,
        run_id="r1",
        generation=generation,
        parent_ids=[],
        x="x",
        y=0.5,
        fitness=0.5,
        problem_id="BBOB" if instance_kind == "bbob" else "MA-BBOB",
        candidate_id=id_,
        instance_index=0,
        instance_kind=instance_kind,
        instance_fid_or_idx=instance_fid_or_idx,
    )


def test_leave_function_out_split_raises_on_non_exploded_examples():
    examples = [
        RLMExample(
            id="a", run_id="r", generation=0, parent_ids=[], x="x", y=0.5, fitness=0.5
        )
    ]
    with pytest.raises(ValueError):
        leave_function_out_split(examples, holdout_fids=[1])


def test_leave_function_out_split_holds_out_only_requested_fids():
    examples = [
        _exploded_example("bbob_f1_a", "bbob", 1),
        _exploded_example("bbob_f1_b", "bbob", 1),
        _exploded_example("bbob_f2", "bbob", 2),
        _exploded_example("bbob_f3", "bbob", 3),
    ]
    result = leave_function_out_split(examples, holdout_fids=[2])
    assert {e.id for e in result.test} == {"bbob_f2"}
    train_val_ids = {e.id for e in result.train} | {e.id for e in result.val}
    assert train_val_ids == {"bbob_f1_a", "bbob_f1_b", "bbob_f3"}
    assert result.log["strategy"] == "leave_function_out"
    assert result.log["holdout_fids"] == [2]


def test_leave_function_out_split_never_places_ma_bbob_in_test():
    examples = [
        _exploded_example("bbob_f1", "bbob", 1),
        _exploded_example("ma1", "ma_bbob", 0),
        _exploded_example("ma2", "ma_bbob", 1),
    ]
    # Holding out fid 1 would normally catch bbob_f1, but ma_bbob rows (idx
    # 0/1 in their own CSV-row namespace) must never leak into test just
    # because their fid_or_idx happens to numerically match a holdout fid.
    result = leave_function_out_split(examples, holdout_fids=[0, 1])
    assert {e.id for e in result.test} == {"bbob_f1"}
    assert {e.id for e in result.train + result.val} == {"ma1", "ma2"}


def test_leave_function_out_split_no_examples_lost_or_duplicated():
    examples = [
        _exploded_example(f"bbob_f{fid}_{i}", "bbob", fid)
        for fid in (1, 2, 3, 4)
        for i in range(3)
    ]
    result = leave_function_out_split(examples, holdout_fids=[3, 4], val_fraction=0.2)
    all_ids = [e.id for e in result.train + result.val + result.test]
    assert len(all_ids) == len(examples)
    assert len(set(all_ids)) == len(all_ids)
    assert all(e.instance_fid_or_idx in (3, 4) for e in result.test)


# --- max_records subsampling ---


def test_run_pipeline_max_records_subsamples_deterministically(tmp_path):
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    summary_a = run_pipeline(
        FIXTURES, out_a, max_records=10, split_config=SplitConfig(seed=0)
    )
    summary_b = run_pipeline(
        FIXTURES, out_b, max_records=10, split_config=SplitConfig(seed=0)
    )
    assert summary_a["n_records_sampled"] == 10
    assert summary_a["n_records_total"] == 98  # full fixture set, unaffected
    assert summary_a["n_records_total"] > summary_a["n_records_sampled"]
    # Same seed -> same subsample -> identical resulting example ids.
    train_a = read_examples_jsonl(out_a / "train.jsonl")
    train_b = read_examples_jsonl(out_b / "train.jsonl")
    assert [e.id for e in train_a] == [e.id for e in train_b]


def test_run_pipeline_max_records_noop_when_larger_than_dataset(tmp_path):
    summary = run_pipeline(FIXTURES, tmp_path / "out", max_records=10_000)
    assert summary["n_records_sampled"] == summary["n_records_total"] == 98
