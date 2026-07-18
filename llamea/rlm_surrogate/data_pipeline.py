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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .schema import BladeRecord, derive_run_id, iter_blade_records, validate_records

TargetKind = Literal["fitness", "aucs"]


@dataclass
class RLMExample:
    """One (x, y) training example, plus lineage metadata needed for
    generation/lineage-aware splitting and later per-run evaluation."""

    id: str
    run_id: str
    generation: int
    parent_ids: list[str]
    x: str
    y: float | list[float]
    fitness: float  # raw scalar fitness, always kept for reporting/eval

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
    """Drops records with a non-empty ``error`` -- their fitness isn't meaningful."""
    kept = [r for r in records if not r.has_error]
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
            )
        )
    return examples


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


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
) -> dict[str, Any]:
    """Runs the full Step 1 pipeline and writes train/val/test + stats to
    ``output_dir``. Returns the summary dict that also gets written out."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")

    per_file_stats = []
    all_records: list[BladeRecord] = []
    all_kept: list[BladeRecord] = []
    for f in files:
        run_id = derive_run_id(f)
        records = list(iter_blade_records(f))
        kept, n_dropped = filter_errored(records)
        report = validate_records(records, label=run_id)
        per_file_stats.append(report.to_dict())
        all_records.extend(records)
        all_kept.extend(kept)

    overall_report = validate_records(all_records, label="__overall__")

    examples = make_examples(
        all_kept,
        include_description=include_description,
        include_configspace=include_configspace,
        target=target,
    )
    n_skipped_missing_target = len(all_kept) - len(examples)

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
        "n_records_total": len(all_records),
        "n_errored_total": len(all_records) - len(all_kept),
        "error_fraction_total": (
            (len(all_records) - len(all_kept)) / len(all_records)
            if all_records
            else 0.0
        ),
        "n_skipped_missing_target": n_skipped_missing_target,
        "n_examples_total": len(examples),
        "per_file": per_file_stats,
        "overall": overall_report.to_dict(),
        "split": split.log,
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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir", required=True, help="Directory of BLADE-schema .jsonl logs."
    )
    p.add_argument(
        "--output-dir", required=True, help="Where to write train/val/test/stats."
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
    p.add_argument("--target", choices=["fitness", "aucs"], default="fitness")
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
    summary = run_pipeline(
        args.data_dir,
        args.output_dir,
        include_description=args.include_description,
        include_configspace=args.include_configspace,
        target=args.target,
        split_config=split_config,
        pattern=args.pattern,
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
