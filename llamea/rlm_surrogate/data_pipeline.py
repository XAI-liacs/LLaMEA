"""BLADE jsonl logs -> RLM-ready (x, y) training examples.

CLI:
    python -m llamea.rlm_surrogate.data_pipeline \\
        --data-dir /path/to/logs --output-dir /path/to/output

Takes a directory of jsonl files (one file == one run/session) as required
input -- never a single hardcoded path, since the real data directory does
not exist yet and will be supplied later.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .schema import BladeRecord, derive_run_id, iter_blade_records, validate_records

TargetKind = Literal["fitness", "aucs", "aucs_per_instance"]

# Folder-name substrings that mark a run as scratch/known-bad rather than a
# real experiment, for the nested per-problem-subdir layout. Excluded by
# default (see `run_pipeline_multi_problem`'s `exclude_dirs`); pass an
# explicit `exclude_dirs=set()` to include everything instead.
DEFAULT_EXCLUDE_DIR_SUBSTRINGS = ("debug", "wrong")


@dataclass
class RLMExample:
    """One (x, y) training example, plus lineage metadata needed for
    generation/lineage-aware splitting and later per-run/per-problem
    evaluation."""

    id: str
    run_id: str
    generation: int
    parent_ids: list[str]
    x: str
    y: float | list[float]
    fitness: float  # raw scalar fitness, always kept for reporting/eval
    problem_id: str = ""

    # Set only when this example was produced by
    # `explode_aucs_with_problem_features` -- one row per `metadata.aucs[i]`
    # rather than one row per candidate. `candidate_id` is the original
    # BladeRecord.id (shared across all of that candidate's exploded rows);
    # empty for non-exploded examples (the common case). See
    # `evaluate.aggregate_instance_predictions` for rolling these back up
    # to one prediction per candidate.
    candidate_id: str = ""
    instance_index: int = -1

    # Also set only when exploded (see above). `instance_kind` is
    # `ProblemInstance.kind` ("bbob"/"ma_bbob"); `instance_fid_or_idx` is
    # `ProblemInstance.fid_or_idx` -- a canonical BBOB function id (1..24)
    # when `instance_kind == "bbob"`, or an MA-BBOB CSV row index (a
    # different namespace, not a function id) when `instance_kind ==
    # "ma_bbob"`. Lets splits/analyses key on the resolved instance without
    # re-parsing it out of the problem-feature text in `x`. See
    # `leave_function_out_split`.
    instance_kind: str = ""
    instance_fid_or_idx: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_x(
    record: BladeRecord,
    *,
    include_description: bool = True,
    include_configspace: bool = True,
) -> str:
    """Builds the input text for a record.

    ``code`` is always the primary/final content. ``description`` and
    ``configspace`` (if non-empty) are optionally prepended as context,
    mirroring the paper's "Problem + Code" vs "Code Only" ablation (Table 14).
    Never includes ``feedback`` or ``fitness`` -- those leak the label.
    """
    parts = []
    if include_description and record.description:
        parts.append(f"# Description\n{record.description.strip()}")
    if include_configspace and record.configspace:
        parts.append(f"# Config space\n{record.configspace.strip()}")
    parts.append(f"# Code\n{record.code}")
    return "\n\n".join(parts)


def build_y(record: BladeRecord, *, target: TargetKind) -> float | list[float] | None:
    if target == "fitness":
        return record.fitness
    if target == "aucs":
        return record.aucs
    raise ValueError(f"Unknown target kind: {target!r}")


def filter_errored(records: list[BladeRecord]) -> tuple[list[BladeRecord], int]:
    """Drops invalid records -- either a non-empty ``error``, or a non-finite
    fitness with an empty ``error`` (some logger versions never populate
    ``error`` on failure; see ``BladeRecord.is_invalid``). Either way the
    fitness isn't meaningful."""
    kept = [r for r in records if not r.is_invalid]
    return kept, len(records) - len(kept)


def make_examples(
    records: list[BladeRecord],
    *,
    include_description: bool = True,
    include_configspace: bool = True,
    target: TargetKind = "fitness",
) -> list[RLMExample]:
    """Converts filtered records into RLM examples, skipping ones with no
    usable target (e.g. ``target="aucs"`` but ``metadata.aucs`` is empty)."""
    examples = []
    for r in records:
        y = build_y(r, target=target)
        if y is None:
            continue
        if isinstance(y, float) and not _is_finite(y):
            continue
        examples.append(
            RLMExample(
                id=r.id,
                run_id=r.run_id,
                generation=r.generation,
                parent_ids=list(r.parent_ids),
                x=build_x(
                    r,
                    include_description=include_description,
                    include_configspace=include_configspace,
                ),
                y=y,
                fitness=r.fitness,
                problem_id=r.problem_id,
            )
        )
    return examples


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


# --------------------------------------------------------------------------
# Per-instance explosion: one example per `metadata.aucs[i]`, with that
# instance's own problem-landscape fingerprint in `x` (see
# `problem_instances.py`). Requires `ioh`.
# --------------------------------------------------------------------------


def explode_aucs_with_problem_features(
    records: list[BladeRecord],
    *,
    include_description: bool = True,
    include_configspace: bool = True,
    n_lhs_points: int = 20,
    lhs_seed: int = 0,
    feature_mode: str = "lhs",
) -> tuple[list[RLMExample], dict[str, int]]:
    """Explodes each record's `metadata.aucs` into one `RLMExample` per
    instance, `x` = code(+context) + that instance's problem-feature text,
    `y` = `aucs[i]`.

    `feature_mode` selects what that problem-feature text contains -- see
    `problem_instances.compute_problem_feature_text` for the full menu
    (`"lhs"` raw samples, `"lhs_stats"` computed summary stats, `"meta"`
    static BBOB/MA-BBOB properties, or a `"meta+..."` combination). Default
    `"lhs"` is unchanged from the original shipped behavior.

    Instance identity comes from real metadata, never a guess --
    `problem_instances.resolve_instances_for_record` tries
    `metadata.performance_data` first, then falls back to the sibling
    `experimentlog.jsonl`. A record is skipped entirely (counted) when it
    has no `aucs`, when neither source resolves an instance list, or when
    the resolved list's length doesn't match `len(aucs)`. Per-instance
    reconstruction failures (e.g. a bad row in the MA-BBOB tables, or an
    unknown fid for `"meta"` mode) are also skipped and counted individually.

    Returns `(examples, counts)` where `counts` has `n_no_aucs`,
    `n_no_instance_mapping`, `n_length_mismatch`, `n_instance_errors`,
    `n_exploded`.
    """
    from .problem_instances import (
        compute_problem_feature_text,
        resolve_instances_for_record,
    )

    examples: list[RLMExample] = []
    counts = {
        "n_no_aucs": 0,
        "n_no_instance_mapping": 0,
        "n_length_mismatch": 0,
        "n_instance_errors": 0,
        "n_exploded": 0,
    }
    experiment_instance_cache: dict[str, Any] = {}

    for r in records:
        aucs = r.aucs
        if not aucs:
            counts["n_no_aucs"] += 1
            continue
        instances = resolve_instances_for_record(r, experiment_instance_cache)
        if instances is None:
            counts["n_no_instance_mapping"] += 1
            continue
        if len(instances) != len(aucs):
            counts["n_length_mismatch"] += 1
            continue

        base_x = build_x(
            r,
            include_description=include_description,
            include_configspace=include_configspace,
        )
        for i, (y, instance) in enumerate(zip(aucs, instances)):
            if not _is_finite(y):
                continue
            try:
                feature_text = compute_problem_feature_text(
                    instance, n_points=n_lhs_points, seed=lhs_seed, mode=feature_mode
                )
            except Exception:
                counts["n_instance_errors"] += 1
                continue
            examples.append(
                RLMExample(
                    id=f"{r.id}#{i}",
                    run_id=r.run_id,
                    generation=r.generation,
                    parent_ids=list(r.parent_ids),
                    x=f"{base_x}\n\n# Problem\n{feature_text}",
                    y=float(y),
                    fitness=r.fitness,
                    problem_id=r.problem_id,
                    candidate_id=r.id,
                    instance_index=i,
                    instance_kind=instance.kind,
                    instance_fid_or_idx=instance.fid_or_idx,
                )
            )
            counts["n_exploded"] += 1

    return examples, counts


def _build_examples(
    records: list[BladeRecord],
    *,
    include_description: bool,
    include_configspace: bool,
    target: TargetKind,
    n_lhs_points: int,
    lhs_seed: int,
    feature_mode: str = "lhs",
) -> tuple[list[RLMExample], dict[str, Any]]:
    """Dispatches to `make_examples` or (for `target="aucs_per_instance"`)
    `explode_aucs_with_problem_features`. Returns `(examples, extra_stats)`
    where `extra_stats` is empty except for the exploded case."""
    if target == "aucs_per_instance":
        examples, counts = explode_aucs_with_problem_features(
            records,
            include_description=include_description,
            include_configspace=include_configspace,
            n_lhs_points=n_lhs_points,
            lhs_seed=lhs_seed,
            feature_mode=feature_mode,
        )
        return examples, {"instance_explosion": counts}

    examples = make_examples(
        records,
        include_description=include_description,
        include_configspace=include_configspace,
        target=target,
    )
    return examples, {}


# --------------------------------------------------------------------------
# Lineage- and generation-aware splitting.
# --------------------------------------------------------------------------


@dataclass
class SplitConfig:
    """Controls how train/val/test are carved out.

    Never a random i.i.d. split: entire runs (files) are preferred as the
    test set when enough runs are available; otherwise (or for the
    validation set) later generations within a run are held out instead of
    randomly interspersing generations, so that near-duplicate mutated
    siblings of a training example don't leak into val/test.
    """

    test_run_fraction: float = 0.2
    min_runs_for_file_holdout: int = 3
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0


@dataclass
class SplitResult:
    train: list[RLMExample]
    val: list[RLMExample]
    test: list[RLMExample]
    log: dict[str, Any] = field(default_factory=dict)


def _generation_holdout(
    examples: list[RLMExample], holdout_fraction: float
) -> tuple[list[RLMExample], list[RLMExample], dict[str, int]]:
    """Within each run, holds out the latest generations until roughly
    ``holdout_fraction`` of that run's examples are held out. Returns
    (kept, held_out, {run_id: cutoff_generation_used_for_holdout})."""
    by_run: dict[str, list[RLMExample]] = {}
    for e in examples:
        by_run.setdefault(e.run_id, []).append(e)

    kept: list[RLMExample] = []
    held_out: list[RLMExample] = []
    cutoffs: dict[str, int] = {}

    for run_id, run_examples in by_run.items():
        if holdout_fraction <= 0:
            kept.extend(run_examples)
            continue
        gens_desc = sorted({e.generation for e in run_examples}, reverse=True)
        target_n = round(len(run_examples) * holdout_fraction)
        held_gens: set[int] = set()
        n_held = 0
        for g in gens_desc:
            if n_held >= target_n and held_gens:
                break
            held_gens.add(g)
            n_held += sum(1 for e in run_examples if e.generation == g)
        # Never hold out an entire single-generation run -- nothing left to train on.
        if len(held_gens) >= len(gens_desc):
            held_gens = set(gens_desc[: max(0, len(gens_desc) - 1)])

        cutoffs[run_id] = min(held_gens) if held_gens else -1
        for e in run_examples:
            (held_out if e.generation in held_gens else kept).append(e)

    return kept, held_out, cutoffs


def lineage_generation_split(
    examples: list[RLMExample], config: SplitConfig | None = None
) -> SplitResult:
    config = config or SplitConfig()
    runs = sorted({e.run_id for e in examples})
    rng = random.Random(config.seed)
    shuffled_runs = runs[:]
    rng.shuffle(shuffled_runs)

    log: dict[str, Any] = {"n_runs": len(runs), "config": asdict(config)}

    if len(runs) >= config.min_runs_for_file_holdout:
        n_test_runs = max(1, round(len(runs) * config.test_run_fraction))
        test_runs = set(shuffled_runs[:n_test_runs])
        log["strategy"] = "whole_run_holdout_for_test"
        log["test_runs"] = sorted(test_runs)

        test = [e for e in examples if e.run_id in test_runs]
        remaining = [e for e in examples if e.run_id not in test_runs]

        train, val, cutoffs = _generation_holdout(remaining, config.val_fraction)
        log["val_generation_cutoffs"] = cutoffs
    else:
        log["strategy"] = "within_run_generation_holdout"
        train_val, test, test_cutoffs = _generation_holdout(
            examples, config.test_fraction
        )
        remaining_val_fraction = config.val_fraction / max(
            1e-9, 1.0 - config.test_fraction
        )
        train, val, val_cutoffs = _generation_holdout(train_val, remaining_val_fraction)
        log["test_generation_cutoffs"] = test_cutoffs
        log["val_generation_cutoffs"] = val_cutoffs

    log["n_train"] = len(train)
    log["n_val"] = len(val)
    log["n_test"] = len(test)
    return SplitResult(train=train, val=val, test=test, log=log)


def leave_function_out_split(
    examples: list[RLMExample],
    holdout_fids: list[int],
    *,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> SplitResult:
    """Splits so entire BBOB function ids are held out for test -- unlike
    `lineage_generation_split`, this tests whether the model generalizes to
    a landscape it never saw *any* instance of during training, which is
    the actual question problem-feature ablations need answered (a within-
    run/lineage split can't distinguish "learned to use problem features"
    from "memorized this function's score range").

    Requires exploded examples (`target="aucs_per_instance"`, so
    `instance_kind`/`instance_fid_or_idx` are populated) -- raises
    `ValueError` otherwise, since there's nothing to hold out by function on
    plain fitness/aucs examples.

    Only applies to `instance_kind == "bbob"` rows, where `fid_or_idx` is an
    unambiguous canonical function id (1..24). MA-BBOB rows are affine
    combinations of several base functions (see `describe_ma_bbob_composition`
    in `problem_instances.py`) -- "holding out a function" doesn't have a
    clean meaning for a mixture, so all MA-BBOB rows are kept in train/val
    and never placed in test for this split mode.
    """
    if not any(e.instance_kind for e in examples):
        raise ValueError(
            "leave_function_out_split requires exploded examples "
            "(target='aucs_per_instance') -- instance_kind is empty on all "
            "given examples."
        )
    holdout = set(holdout_fids)

    test = [
        e
        for e in examples
        if e.instance_kind == "bbob" and e.instance_fid_or_idx in holdout
    ]
    remaining = [
        e
        for e in examples
        if not (e.instance_kind == "bbob" and e.instance_fid_or_idx in holdout)
    ]
    train, val, val_cutoffs = _generation_holdout(remaining, val_fraction)

    log: dict[str, Any] = {
        "strategy": "leave_function_out",
        "holdout_fids": sorted(holdout),
        "val_generation_cutoffs": val_cutoffs,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    return SplitResult(train=train, val=val, test=test, log=log)


# --------------------------------------------------------------------------
# Stats / IO.
# --------------------------------------------------------------------------


def write_examples_jsonl(examples: list[RLMExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for e in examples:
            fh.write(json.dumps(e.to_dict()) + "\n")


def read_examples_jsonl(path: str | Path) -> list[RLMExample]:
    examples = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(RLMExample(**d))
    return examples


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    include_description: bool = True,
    include_configspace: bool = True,
    target: TargetKind = "fitness",
    split_config: SplitConfig | None = None,
    pattern: str = "*.jsonl",
    n_lhs_points: int = 20,
    lhs_seed: int = 0,
    feature_mode: str = "lhs",
    holdout_fids: list[int] | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Runs the full Step 1 pipeline and writes train/val/test + stats to
    ``output_dir``. Returns the summary dict that also gets written out.

    ``max_records``, if given, randomly subsamples (seeded by
    ``split_config.seed``) down to that many records right after loading --
    for a fast ablation/sanity pass over a large dataset, not for the real
    run. ``holdout_fids``, if given (only meaningful with
    ``target="aucs_per_instance"``), switches the split to
    ``leave_function_out_split`` instead of ``lineage_generation_split``.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")

    per_file_stats = []
    all_records: list[BladeRecord] = []
    for f in files:
        run_id = derive_run_id(f)
        records = list(iter_blade_records(f))
        report = validate_records(records, label=run_id)
        per_file_stats.append(report.to_dict())
        all_records.extend(records)

    overall_report = validate_records(all_records, label="__overall__")

    sample_records = all_records
    if max_records is not None and len(all_records) > max_records:
        sample_records = random.Random((split_config or SplitConfig()).seed).sample(
            all_records, max_records
        )
    all_kept, _ = filter_errored(sample_records)

    examples, extra_stats = _build_examples(
        all_kept,
        include_description=include_description,
        include_configspace=include_configspace,
        target=target,
        n_lhs_points=n_lhs_points,
        lhs_seed=lhs_seed,
        feature_mode=feature_mode,
    )
    if "instance_explosion" in extra_stats:
        counts = extra_stats["instance_explosion"]
        n_skipped_missing_target = (
            counts["n_no_aucs"]
            + counts["n_no_instance_mapping"]
            + counts["n_length_mismatch"]
        )
    else:
        n_skipped_missing_target = len(all_kept) - len(examples)

    if holdout_fids:
        split = leave_function_out_split(
            examples,
            holdout_fids,
            val_fraction=(split_config or SplitConfig()).val_fraction,
            seed=(split_config or SplitConfig()).seed,
        )
    else:
        split = lineage_generation_split(examples, split_config)

    write_examples_jsonl(split.train, output_dir / "train.jsonl")
    write_examples_jsonl(split.val, output_dir / "val.jsonl")
    write_examples_jsonl(split.test, output_dir / "test.jsonl")

    summary = {
        "n_files": len(files),
        "files": [str(f) for f in files],
        "target": target,
        "include_description": include_description,
        "include_configspace": include_configspace,
        "max_records": max_records,
        "n_records_total": len(all_records),
        "n_records_sampled": len(sample_records),
        "n_errored_total": len(sample_records) - len(all_kept),
        "error_fraction_total": (
            (len(sample_records) - len(all_kept)) / len(sample_records)
            if sample_records
            else 0.0
        ),
        "n_skipped_missing_target": n_skipped_missing_target,
        "n_examples_total": len(examples),
        "per_file": per_file_stats,
        "overall": overall_report.to_dict(),
        "split": split.log,
        **extra_stats,
    }

    if len(examples) < 1000:
        summary.setdefault("warnings", []).append(
            f"Only {len(examples)} usable examples across {len(files)} file(s). "
            "The RLM paper's strongest results come from tens-of-thousands to "
            "millions of examples; at this scale, treat correlation numbers as "
            "a few-shot/fine-tuning signal, not a trained-model guarantee, and "
            "prefer pooling multiple runs/files before trusting them."
        )

    with open(output_dir / "stats.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


# --------------------------------------------------------------------------
# Nested `<experiment_folder>/run-*/log.jsonl` layout (real BLADE-results
# exports), with `problem_id` derived from the experiment folder name.
# --------------------------------------------------------------------------


def classify_problem(
    folder_name: str, problem_map: dict[str, str] | None = None
) -> str:
    """Maps an experiment-folder name to a problem_id.

    ``problem_map`` (folder name -> problem_id) takes precedence when given.
    Otherwise, folders starting with "MA-BBOB"/"MABBOB"/"MA_BBOB" (any
    case/separator) are grouped as "MA-BBOB", any other folder containing
    "BBOB" is grouped as "BBOB", and anything else falls back to using its
    own folder name as a singleton problem_id (surfaced, not silently
    dropped, so an unrecognized folder doesn't get mis-grouped)."""
    if problem_map is not None and folder_name in problem_map:
        return problem_map[folder_name]
    if re.match(r"^MA[-_]?BBOB", folder_name, re.IGNORECASE):
        return "MA-BBOB"
    if "BBOB" in folder_name.upper():
        return "BBOB"
    return folder_name


def load_nested_directory(
    root_dir: str | Path,
    *,
    exclude_dir_substrings: tuple[str, ...] = DEFAULT_EXCLUDE_DIR_SUBSTRINGS,
    problem_map: dict[str, str] | None = None,
) -> tuple[list[BladeRecord], list[str]]:
    """Loads a `<experiment_folder>/run-*/log.jsonl` tree (as produced by
    ``llamea.loggers.ExperimentLogger`` -- ``conversationlog.jsonl``,
    ``experimentlog.jsonl``, and ``progress.json`` are intentionally
    ignored, they aren't per-candidate BLADE records).

    ``problem_id`` is derived per experiment folder via ``classify_problem``;
    ``run_id`` is ``f"{experiment_folder}/{run_folder}"`` so it stays
    globally unique even though every run folder contains a same-named
    ``log.jsonl``. Experiment folders whose name contains any of
    ``exclude_dir_substrings`` (case-insensitive) are skipped -- default
    excludes anything with "debug" or "wrong" in the name, since those
    folders showed up as scratch/known-bad runs in practice; pass
    ``exclude_dir_substrings=()`` to include everything.

    Returns ``(records, excluded_experiment_folder_names)``.
    """
    root_dir = Path(root_dir)
    records: list[BladeRecord] = []
    excluded: list[str] = []
    for exp_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        if any(s.lower() in exp_dir.name.lower() for s in exclude_dir_substrings):
            excluded.append(exp_dir.name)
            continue
        problem_id = classify_problem(exp_dir.name, problem_map)
        for run_dir in sorted(exp_dir.glob("run-*")):
            log_file = run_dir / "log.jsonl"
            if not log_file.exists():
                continue
            run_id = f"{exp_dir.name}/{run_dir.name}"
            records.extend(
                iter_blade_records(log_file, run_id=run_id, problem_id=problem_id)
            )
    return records, excluded


def run_pipeline_multi_problem(
    root_dir: str | Path,
    output_dir: str | Path,
    *,
    include_description: bool = True,
    include_configspace: bool = True,
    target: TargetKind = "fitness",
    split_config: SplitConfig | None = None,
    exclude_dir_substrings: tuple[str, ...] = DEFAULT_EXCLUDE_DIR_SUBSTRINGS,
    problem_map: dict[str, str] | None = None,
    n_lhs_points: int = 20,
    lhs_seed: int = 0,
    feature_mode: str = "lhs",
    holdout_fids: list[int] | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Like ``run_pipeline``, but for the nested per-experiment-folder
    layout, with the train/val/test split run independently per
    ``problem_id`` and then merged -- so whole-run test-holdout and
    generation-based val-holdout are each stratified by problem instead of
    one problem's runs dominating the held-out set.

    ``max_records``/``holdout_fids``: see ``run_pipeline``. When
    ``holdout_fids`` is given, the per-problem split loop is skipped in
    favor of one ``leave_function_out_split`` call over all problems'
    examples together (it already handles BBOB vs MA-BBOB rows correctly).
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    all_records, excluded_dirs = load_nested_directory(
        root_dir,
        exclude_dir_substrings=exclude_dir_substrings,
        problem_map=problem_map,
    )
    if not all_records:
        raise FileNotFoundError(
            f"No `<experiment>/run-*/log.jsonl` files found under {root_dir} "
            f"(excluded folders: {excluded_dirs})"
        )

    by_source_file: dict[str, list[BladeRecord]] = {}
    for r in all_records:
        by_source_file.setdefault(r.source_file, []).append(r)
    per_file_stats = [
        validate_records(recs, label=recs[0].run_id).to_dict()
        for _, recs in sorted(by_source_file.items())
    ]

    overall_report = validate_records(all_records, label="__overall__")

    sample_records = all_records
    if max_records is not None and len(all_records) > max_records:
        sample_records = random.Random((split_config or SplitConfig()).seed).sample(
            all_records, max_records
        )

    kept, n_dropped = filter_errored(sample_records)
    examples, extra_stats = _build_examples(
        kept,
        include_description=include_description,
        include_configspace=include_configspace,
        target=target,
        n_lhs_points=n_lhs_points,
        lhs_seed=lhs_seed,
        feature_mode=feature_mode,
    )
    if "instance_explosion" in extra_stats:
        counts = extra_stats["instance_explosion"]
        n_skipped_missing_target = (
            counts["n_no_aucs"]
            + counts["n_no_instance_mapping"]
            + counts["n_length_mismatch"]
        )
    else:
        n_skipped_missing_target = len(kept) - len(examples)

    by_problem: dict[str, list[RLMExample]] = {}
    for e in examples:
        by_problem.setdefault(e.problem_id, []).append(e)

    if holdout_fids:
        result = leave_function_out_split(
            examples,
            holdout_fids,
            val_fraction=(split_config or SplitConfig()).val_fraction,
            seed=(split_config or SplitConfig()).seed,
        )
        train, val, test = result.train, result.val, result.test
        split_log_by_problem = {"__all_problems__": result.log}
    else:
        train, val, test = [], [], []
        split_log_by_problem = {}
        for problem_id, problem_examples in sorted(by_problem.items()):
            result = lineage_generation_split(problem_examples, split_config)
            train.extend(result.train)
            val.extend(result.val)
            test.extend(result.test)
            split_log_by_problem[problem_id] = result.log

    write_examples_jsonl(train, output_dir / "train.jsonl")
    write_examples_jsonl(val, output_dir / "val.jsonl")
    write_examples_jsonl(test, output_dir / "test.jsonl")

    per_problem_stats = {
        problem_id: validate_records(
            [r for r in all_records if r.problem_id == problem_id],
            label=problem_id,
        ).to_dict()
        for problem_id in sorted(by_problem)
    }

    strategies = sorted(
        {log.get("strategy", "") for log in split_log_by_problem.values()}
    )
    summary: dict[str, Any] = {
        "layout": "per_problem_subdir",
        "root_dir": str(root_dir),
        "excluded_experiment_folders": excluded_dirs,
        "problems": sorted(by_problem),
        "target": target,
        "include_description": include_description,
        "include_configspace": include_configspace,
        "max_records": max_records,
        "n_records_total": len(all_records),
        "n_records_sampled": len(sample_records),
        "n_errored_total": n_dropped,
        "error_fraction_total": (
            (n_dropped / len(sample_records)) if sample_records else 0.0
        ),
        "n_skipped_missing_target": n_skipped_missing_target,
        "n_examples_total": len(examples),
        "per_file": per_file_stats,
        "per_problem": per_problem_stats,
        "overall": overall_report.to_dict(),
        "split": {
            "strategy": "+".join(strategies),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "by_problem": split_log_by_problem,
            "test_runs": sorted(
                r
                for log in split_log_by_problem.values()
                for r in log.get("test_runs", [])
            ),
        },
        **extra_stats,
    }

    if len(examples) < 1000:
        summary.setdefault("warnings", []).append(
            f"Only {len(examples)} usable examples across "
            f"{len(by_problem)} problem(s). The RLM paper's strongest "
            "results come from tens-of-thousands to millions of examples; "
            "at this scale, treat correlation numbers as a few-shot/"
            "fine-tuning signal, not a trained-model guarantee."
        )

    with open(output_dir / "stats.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory of BLADE-schema .jsonl logs (--layout flat), or the "
        "root containing one experiment-folder-per-run-group "
        "(--layout per_problem_subdir).",
    )
    p.add_argument(
        "--output-dir", required=True, help="Where to write train/val/test/stats."
    )
    p.add_argument(
        "--layout",
        choices=["flat", "per_problem_subdir"],
        default="flat",
        help="'flat': --data-dir directly contains .jsonl run files. "
        "'per_problem_subdir': --data-dir contains "
        "<experiment_folder>/run-*/log.jsonl (problem_id derived from the "
        "experiment folder name).",
    )
    p.add_argument(
        "--exclude-dir-substring",
        action="append",
        default=None,
        dest="exclude_dir_substrings",
        help="(per_problem_subdir only) Skip experiment folders whose name "
        "contains this substring, case-insensitive. Repeatable. Defaults to "
        f"{DEFAULT_EXCLUDE_DIR_SUBSTRINGS}; pass this flag with an empty "
        "string once to include everything.",
    )
    p.add_argument("--pattern", default="*.jsonl")
    p.add_argument(
        "--no-description",
        dest="include_description",
        action="store_false",
        help="Exclude the description field from x (code-only ablation).",
    )
    p.add_argument(
        "--no-configspace",
        dest="include_configspace",
        action="store_false",
        help="Exclude the configspace field from x.",
    )
    p.add_argument(
        "--target",
        choices=["fitness", "aucs", "aucs_per_instance"],
        default="fitness",
        help="'fitness': scalar aggregate score. 'aucs': the whole "
        "metadata.aucs vector as one multi-objective example. "
        "'aucs_per_instance': explode each record into one example per "
        "metadata.aucs[i], x augmented with that instance's own Latin "
        "Hypercube problem fingerprint (see problem_instances.py). Instance "
        "identity comes from metadata.performance_data or the sibling "
        "experimentlog.jsonl -- records where neither resolves are skipped, "
        "not guessed. Requires `ioh`.",
    )
    p.add_argument(
        "--lhs-points",
        type=int,
        default=20,
        dest="n_lhs_points",
        help="(aucs_per_instance only) Number of Latin Hypercube sample "
        "points per problem fingerprint.",
    )
    p.add_argument(
        "--lhs-seed",
        type=int,
        default=0,
        help="(aucs_per_instance only) Seed for the LHS sampler -- fixed "
        "across instances of the same dimensionality so fingerprints share "
        "probe locations and stay comparable.",
    )
    p.add_argument(
        "--feature-mode",
        choices=["lhs", "lhs_stats", "meta", "meta+lhs", "meta+lhs_stats"],
        default="lhs",
        help="(aucs_per_instance only) What problem-feature text to append: "
        "'lhs' (default, unchanged): raw Latin Hypercube samples. "
        "'lhs_stats': computed summary statistics from the same LHS sample "
        "instead of raw text. 'meta': static BBOB/MA-BBOB properties only "
        "(no function evaluations, no `ioh` needed). 'meta+lhs'/"
        "'meta+lhs_stats': both combined. See problem_instances.py.",
    )
    p.add_argument(
        "--holdout-fids",
        type=int,
        nargs="+",
        default=None,
        help="(aucs_per_instance only) Canonical BBOB function ids (1..24) "
        "to hold out entirely for test -- switches from the default "
        "lineage/generation split to leave_function_out_split, so test "
        "contains only instances of these functions and train/val contain "
        "none. For measuring cross-problem generalization, not for the "
        "default run.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Randomly subsample down to this many records (seeded by "
        "--seed) right after loading, before filtering/exploding. For a "
        "fast ablation/sanity pass over a large dataset -- omit for the "
        "real run.",
    )
    p.add_argument("--test-run-fraction", type=float, default=0.2)
    p.add_argument("--min-runs-for-file-holdout", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(include_description=True, include_configspace=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    split_config = SplitConfig(
        test_run_fraction=args.test_run_fraction,
        min_runs_for_file_holdout=args.min_runs_for_file_holdout,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    if args.layout == "per_problem_subdir":
        exclude_dir_substrings = (
            tuple(args.exclude_dir_substrings)
            if args.exclude_dir_substrings is not None
            else DEFAULT_EXCLUDE_DIR_SUBSTRINGS
        )
        summary = run_pipeline_multi_problem(
            args.data_dir,
            args.output_dir,
            include_description=args.include_description,
            include_configspace=args.include_configspace,
            target=args.target,
            split_config=split_config,
            exclude_dir_substrings=exclude_dir_substrings,
            n_lhs_points=args.n_lhs_points,
            lhs_seed=args.lhs_seed,
            feature_mode=args.feature_mode,
            holdout_fids=args.holdout_fids,
            max_records=args.max_records,
        )
        print(
            f"Ingested {summary['n_records_total']} records for problems "
            f"{summary['problems']} (excluded folders: "
            f"{summary['excluded_experiment_folders']}); dropped "
            f"{summary['n_errored_total']} ({summary['error_fraction_total']:.1%}) "
            f"as invalid; {summary['n_examples_total']} usable examples -> "
            f"train={summary['split']['n_train']} val={summary['split']['n_val']} "
            f"test={summary['split']['n_test']} "
            f"(strategy={summary['split']['strategy']})."
        )
    else:
        summary = run_pipeline(
            args.data_dir,
            args.output_dir,
            include_description=args.include_description,
            include_configspace=args.include_configspace,
            target=args.target,
            split_config=split_config,
            pattern=args.pattern,
            n_lhs_points=args.n_lhs_points,
            lhs_seed=args.lhs_seed,
            feature_mode=args.feature_mode,
            holdout_fids=args.holdout_fids,
            max_records=args.max_records,
        )
        print(
            f"Ingested {summary['n_records_total']} records from {summary['n_files']} "
            f"file(s); dropped {summary['n_errored_total']} "
            f"({summary['error_fraction_total']:.1%}) for errors; "
            f"{summary['n_examples_total']} usable examples -> "
            f"train={summary['split']['n_train']} val={summary['split']['n_val']} "
            f"test={summary['split']['n_test']} "
            f"(strategy={summary['split']['strategy']})."
        )
    for w in summary.get("warnings", []):
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
