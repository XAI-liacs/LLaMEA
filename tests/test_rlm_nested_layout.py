import json
import math
from pathlib import Path

import pytest

from llamea.rlm_surrogate.data_pipeline import (
    classify_problem,
    load_nested_directory,
    run_pipeline_multi_problem,
)
from llamea.rlm_surrogate.schema import BladeRecord


def _record(id_, fitness, error="", feedback=""):
    return {
        "id": id_,
        "fitness": fitness,
        "name": f"Algo{id_}",
        "description": "A synthetic optimizer.",
        "code": f"class Algo{id_}:\n    def __call__(self, func):\n        return {fitness}\n",
        "configspace": "",
        "generation": 0,
        "feedback": feedback,
        "error": error,
        "parent_ids": [],
        "operator": None,
        "metadata": {"aucs": [fitness, fitness]} if error == "" else {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture()
def nested_root(tmp_path):
    root = tmp_path / "BLADE-results"

    # BBOB_exp1: two runs, all healthy records.
    for run in ("run-a", "run-b"):
        recs = [_record(f"bbob-{run}-{i}", 0.5 + 0.01 * i) for i in range(6)]
        _write_jsonl(root / "BBOB_exp1" / run / "log.jsonl", recs)
        # conversationlog.jsonl must be ignored by the loader.
        _write_jsonl(
            root / "BBOB_exp1" / run / "conversationlog.jsonl",
            [{"role": "user", "content": "not a candidate record"}],
        )

    # MA_BBOB_exp1: one run with a real error AND an MA_BBOB-style silent
    # error (empty `error`, non-finite fitness, exception-y feedback).
    ma_recs = [_record(f"ma-{i}", 0.4 + 0.01 * i) for i in range(5)]
    ma_recs.append(_record("ma-real-error", 0.0, error="ValueError: boom"))
    ma_recs.append(
        _record(
            "ma-silent-error",
            float("-inf"),
            error="",
            feedback="An exception occurred: operands could not be broadcast together",
        )
    )
    _write_jsonl(root / "MA_BBOB_exp1" / "run-c" / "log.jsonl", ma_recs)

    # MA_BBOB_debug: should be excluded by default (name contains "debug").
    _write_jsonl(
        root / "MA_BBOB_debug" / "run-d" / "log.jsonl",
        [_record("debug-1", 0.9)],
    )

    return root


def test_classify_problem_groups_ma_bbob_vs_bbob():
    assert classify_problem("MA-BBOB-NEW") == "MA-BBOB"
    assert classify_problem("MABBOB_guided_3") == "MA-BBOB"
    assert classify_problem("MA_BBOB") == "MA-BBOB"
    assert classify_problem("BBOB_guided4") == "BBOB"
    assert classify_problem("BBOB-BO2") == "BBOB"
    assert classify_problem("something_else") == "something_else"


def test_classify_problem_respects_explicit_map():
    assert classify_problem("BBOB_guided4", {"BBOB_guided4": "custom"}) == "custom"


def test_load_nested_directory_ignores_non_log_files_and_excludes_debug(nested_root):
    records, excluded = load_nested_directory(nested_root)
    assert excluded == ["MA_BBOB_debug"]
    # 2 BBOB runs * 6 + 1 MA_BBOB run * 7 = 19 records; conversationlog and
    # the excluded debug folder must not contribute any.
    assert len(records) == 19
    assert all(r.run_id.count("/") == 1 for r in records)  # "<exp>/<run>"


def test_load_nested_directory_derives_problem_id(nested_root):
    records, _ = load_nested_directory(nested_root)
    problems = {r.problem_id for r in records}
    assert problems == {"BBOB", "MA-BBOB"}


def test_load_nested_directory_include_everything_with_empty_exclude(nested_root):
    records, excluded = load_nested_directory(nested_root, exclude_dir_substrings=())
    assert excluded == []
    assert len(records) == 20


def test_ma_bbob_silent_error_is_flagged_invalid(nested_root):
    records, _ = load_nested_directory(nested_root)
    silent = next(r for r in records if r.id == "ma-silent-error")
    assert not silent.has_error  # error field genuinely empty
    assert silent.is_invalid  # but caught via non-finite fitness
    real_error = next(r for r in records if r.id == "ma-real-error")
    assert real_error.has_error
    assert real_error.is_invalid


def test_run_pipeline_multi_problem_end_to_end(nested_root, tmp_path):
    out_dir = tmp_path / "out"
    summary = run_pipeline_multi_problem(nested_root, out_dir)

    assert summary["layout"] == "per_problem_subdir"
    assert summary["excluded_experiment_folders"] == ["MA_BBOB_debug"]
    assert set(summary["problems"]) == {"BBOB", "MA-BBOB"}
    # 2 real errors dropped (real-error + silent-error) out of 19 records.
    assert summary["n_errored_total"] == 2
    assert summary["n_records_total"] == 19
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()

    from llamea.rlm_surrogate.data_pipeline import read_examples_jsonl

    all_examples = (
        read_examples_jsonl(out_dir / "train.jsonl")
        + read_examples_jsonl(out_dir / "val.jsonl")
        + read_examples_jsonl(out_dir / "test.jsonl")
    )
    assert len(all_examples) == summary["n_examples_total"]
    assert all(e.problem_id in ("BBOB", "MA-BBOB") for e in all_examples)
    # No record with the dropped ids should have survived into any split.
    assert "ma-real-error" not in {e.id for e in all_examples}
    assert "ma-silent-error" not in {e.id for e in all_examples}


def test_run_pipeline_multi_problem_raises_when_everything_excluded(tmp_path):
    root = tmp_path / "onlydebug"
    _write_jsonl(root / "BBOB_debug" / "run-x" / "log.jsonl", [_record("x", 0.5)])
    with pytest.raises(FileNotFoundError):
        run_pipeline_multi_problem(root, tmp_path / "out")


# --- aucs_per_instance over the real nested layout (needs the `ioh` extra) ---

ioh = pytest.importorskip("ioh")


def _record_with_performance_data(id_, aucs, fid_iid_pairs, dim=2):
    r = _record(id_, sum(aucs) / len(aucs))
    r["metadata"] = {
        "aucs": aucs,
        "performance_data": [
            {"fid": fid, "iid": iid, "dim": dim, "auc": auc}
            for (fid, iid), auc in zip(fid_iid_pairs, aucs)
        ],
    }
    return r


@pytest.fixture()
def nested_root_with_instance_metadata(tmp_path):
    """Mirrors the real `<experiment>/run-*/log.jsonl` +
    `experimentlog.jsonl` layout, with a BBOB experiment resolving instances
    via each record's own `metadata.performance_data`, and an MA_BBOB
    experiment relying on the sibling `experimentlog.jsonl` fallback --
    exactly the two real resolution paths confirmed against real
    BLADE-results logs."""
    root = tmp_path / "BLADE-results"

    bbob_run = root / "BBOB_exp1" / "run-a"
    _write_jsonl(
        bbob_run / "log.jsonl",
        [
            _record_with_performance_data(f"bbob-{i}", [0.5, 0.6], [(1, 1), (1, 2)])
            for i in range(3)
        ],
    )

    ma_run_dir = root / "MA_BBOB_exp1" / "run-LLaMEA-MA_BBOB-1"
    _write_jsonl(
        ma_run_dir / "log.jsonl",
        [_record(f"ma-{i}", 0.4 + 0.01 * i) for i in range(3)],
    )
    _write_jsonl(
        root / "MA_BBOB_exp1" / "experimentlog.jsonl",
        [
            {
                "log_dir": "run-LLaMEA-MA_BBOB-1",
                "problem": {
                    "name": "MA_BBOB",
                    "dims": [2],
                    "training_instances": "range(0, 2)",
                },
            }
        ],
    )

    return root


def test_run_pipeline_multi_problem_aucs_per_instance_resolves_both_sources(
    nested_root_with_instance_metadata, tmp_path
):
    out_dir = tmp_path / "out"
    summary = run_pipeline_multi_problem(
        nested_root_with_instance_metadata,
        out_dir,
        target="aucs_per_instance",
        n_lhs_points=2,
    )

    counts = summary["instance_explosion"]
    assert counts["n_no_instance_mapping"] == 0
    assert counts["n_length_mismatch"] == 0
    # 3 BBOB records x 2 instances + 3 MA_BBOB records x 2 instances.
    assert counts["n_exploded"] == 12
    assert summary["n_examples_total"] == 12

    from llamea.rlm_surrogate.data_pipeline import read_examples_jsonl

    all_examples = (
        read_examples_jsonl(out_dir / "train.jsonl")
        + read_examples_jsonl(out_dir / "val.jsonl")
        + read_examples_jsonl(out_dir / "test.jsonl")
    )
    assert len(all_examples) == 12
    assert all("# Problem" in e.x and "LHS[2pts" in e.x for e in all_examples)
    candidate_ids = {e.candidate_id for e in all_examples}
    assert candidate_ids == {
        "bbob-0",
        "bbob-1",
        "bbob-2",
        "ma-0",
        "ma-1",
        "ma-2",
    }
