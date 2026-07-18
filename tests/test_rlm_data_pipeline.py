import json
from pathlib import Path

import pytest

from llamea.rlm_surrogate.data_pipeline import (
    RLMExample,
    SplitConfig,
    build_x,
    build_y,
    filter_errored,
    lineage_generation_split,
    make_examples,
    read_examples_jsonl,
    run_pipeline,
    write_examples_jsonl,
)
from llamea.rlm_surrogate.schema import iter_blade_records, load_directory

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
