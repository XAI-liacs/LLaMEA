import math
from pathlib import Path

from llamea.rlm_surrogate.schema import (
    derive_run_id,
    iter_blade_records,
    load_directory,
    validate_records,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rlm"
MALFORMED_FIXTURES = Path(__file__).parent / "fixtures" / "rlm_malformed"


def test_derive_run_id():
    assert derive_run_id("/a/b/run_alpha.jsonl") == "run_alpha"


def test_iter_blade_records_basic_fields():
    records = list(iter_blade_records(FIXTURES / "run_alpha.jsonl"))
    assert len(records) == 48
    r = records[0]
    assert r.run_id == "run_alpha"
    assert r.id
    assert isinstance(r.parent_ids, list)
    assert r.generation == 0
    assert r.source_file.endswith("run_alpha.jsonl")


def test_iter_blade_records_skips_malformed_lines():
    records = list(iter_blade_records(MALFORMED_FIXTURES / "run_malformed.jsonl"))
    # One valid record; the garbage line and blank line are skipped, not raised.
    assert len(records) == 1
    assert records[0].id == "x1"


def test_load_directory_pools_all_files():
    records = load_directory(FIXTURES)
    run_ids = {r.run_id for r in records}
    assert run_ids == {"run_alpha", "run_beta", "run_gamma"}
    assert len(records) == 48 + 30 + 20


def test_error_records_have_flag_set():
    records = list(iter_blade_records(FIXTURES / "run_alpha.jsonl"))
    errored = [r for r in records if r.has_error]
    assert errored
    assert all(r.has_error for r in errored)


def test_aucs_property():
    records = list(iter_blade_records(FIXTURES / "run_beta.jsonl"))
    ok = [r for r in records if not r.has_error]
    assert all(r.aucs is not None and len(r.aucs) == 5 for r in ok)
    errored = [r for r in records if r.has_error]
    assert all(r.aucs is None for r in errored)


def test_validate_records_reports_error_fraction_and_fitness_range():
    records = list(iter_blade_records(FIXTURES / "run_gamma.jsonl"))
    report = validate_records(records, label="run_gamma")
    assert report.n_records == 20
    assert 0.0 <= report.error_fraction <= 1.0
    assert math.isfinite(report.fitness_min)
    assert math.isfinite(report.fitness_max)
    assert 0.0 <= report.fitness_min <= report.fitness_max <= 1.0


def test_validate_records_flags_high_error_rate():
    from llamea.rlm_surrogate.schema import BladeRecord

    records = [
        BladeRecord(
            id=str(i),
            fitness=0.5,
            name="A",
            description="d",
            code="class A:\n    pass\n",
            configspace="",
            generation=0,
            feedback="",
            error="boom" if i < 8 else "",
            parent_ids=[],
            operator=None,
            metadata={},
            run_id="r",
            source_file="r.jsonl",
            line_no=i,
        )
        for i in range(10)
    ]
    report = validate_records(records, label="r")
    assert report.error_fraction == 0.8
    assert any("error rate" in w for w in report.warnings)


def test_validate_records_empty_input():
    report = validate_records([], label="empty")
    assert report.n_records == 0
    assert report.warnings
