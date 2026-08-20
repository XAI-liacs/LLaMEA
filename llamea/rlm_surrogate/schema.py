"""Parsing and validation for BLADE-schema jsonl logs.

Each line of a BLADE log (one file == one LLaMEA/BLADE run/session, produced
by ``llamea.loggers.ExperimentLogger.log_individual``) is a JSON object with
the fields documented in ``BladeRecord``. There is no explicit run-id field,
so it is derived from the source filename.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REQUIRED_FIELDS = (
    "id",
    "fitness",
    "name",
    "description",
    "code",
    "generation",
    "feedback",
    "error",
    "parent_ids",
    "operator",
)


@dataclass
class BladeRecord:
    """One evaluated candidate from a BLADE/LLaMEA jsonl log."""

    id: str
    fitness: float
    name: str
    description: str
    code: str
    configspace: str
    generation: int
    feedback: str
    error: str
    parent_ids: list[str]
    operator: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Derived, not part of the raw schema.
    run_id: str = ""
    problem_id: str = ""
    source_file: str = ""
    line_no: int = -1

    @property
    def aucs(self) -> list[float] | None:
        aucs = self.metadata.get("aucs") if self.metadata else None
        if not aucs:
            return None
        return [float(a) for a in aucs]

    @property
    def has_error(self) -> bool:
        return bool(self.error)

    @property
    def is_invalid(self) -> bool:
        """True if this record's fitness isn't meaningful and it should be
        dropped -- either an explicit ``error`` string, OR a non-finite
        fitness with no ``error`` set. The latter shows up in real data (a
        logger convention that records a failure via ``feedback``/an
        inf/-inf ``fitness`` sentinel without ever populating ``error``);
        relying on ``error`` alone would silently undercount failures for
        those files."""
        return self.has_error or not math.isfinite(self.fitness)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        run_id: str,
        source_file: str,
        line_no: int,
        problem_id: str = "",
    ) -> "BladeRecord":
        return cls(
            id=str(d.get("id", "")),
            fitness=_coerce_float(d.get("fitness")),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            code=str(d.get("code", "")),
            configspace=_coerce_configspace(d.get("configspace", "")),
            generation=int(d.get("generation", 0) or 0),
            feedback=str(d.get("feedback", "")),
            error=str(d.get("error", "") or ""),
            parent_ids=list(d.get("parent_ids") or []),
            operator=d.get("operator"),
            metadata=dict(d.get("metadata") or {}),
            run_id=run_id,
            problem_id=problem_id,
            source_file=source_file,
            line_no=line_no,
        )


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_configspace(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def derive_run_id(file_path: str | Path) -> str:
    """Derives a run id from a log filename (no explicit run-id field exists)."""
    return Path(file_path).stem


def iter_blade_records(
    file_path: str | Path,
    *,
    run_id: str | None = None,
    problem_id: str = "",
) -> Iterator[BladeRecord]:
    """Yields ``BladeRecord``s from one jsonl file, skipping malformed lines.

    ``run_id`` defaults to the filename stem; pass an explicit value (e.g.
    ``f"{experiment_folder}/{run_folder}"``) when ingesting a nested
    per-problem/per-run directory layout where filenames alone (``log.jsonl``
    everywhere) wouldn't be unique.
    """
    path = Path(file_path)
    run_id = run_id if run_id is not None else derive_run_id(path)
    with open(path, "r") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield BladeRecord.from_dict(
                d,
                run_id=run_id,
                source_file=str(path),
                line_no=line_no,
                problem_id=problem_id,
            )


def load_directory(data_dir: str | Path, pattern: str = "*.jsonl") -> list[BladeRecord]:
    """Loads all BLADE records from every jsonl file matching ``pattern``."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    records: list[BladeRecord] = []
    for f in files:
        records.extend(iter_blade_records(f))
    return records


@dataclass
class SchemaReport:
    """Sanity-check results for one ingested file (or the whole dataset)."""

    file: str
    n_records: int
    n_errored: int
    error_fraction: float
    n_missing_fields: int
    fitness_min: float
    fitness_max: float
    fitness_mean: float
    fitness_skew: float
    frac_configspace_populated: float
    frac_operator_populated: float
    frac_initial_population: float
    generation_fitness_spearman: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        return d


def validate_records(records: list[BladeRecord], label: str) -> SchemaReport:
    """Computes summary statistics and flags anomalies for a set of records.

    This does not raise on anomalies -- it surfaces them as warnings so a high
    error rate or an unexpected fitness convention is visible in the report
    rather than silently assumed away.
    """
    warnings: list[str] = []
    n = len(records)
    if n == 0:
        return SchemaReport(
            file=label,
            n_records=0,
            n_errored=0,
            error_fraction=0.0,
            n_missing_fields=0,
            fitness_min=float("nan"),
            fitness_max=float("nan"),
            fitness_mean=float("nan"),
            fitness_skew=float("nan"),
            frac_configspace_populated=0.0,
            frac_operator_populated=0.0,
            frac_initial_population=0.0,
            generation_fitness_spearman=float("nan"),
            warnings=["No records found."],
        )

    n_errored = sum(r.is_invalid for r in records)
    error_fraction = n_errored / n
    n_silent_errored = sum((not r.has_error) and r.is_invalid for r in records)
    if error_fraction > 0.3:
        warnings.append(
            f"High error rate ({error_fraction:.1%}) -- worth investigating "
            "the run itself, not just discarding these records."
        )
    if n_silent_errored:
        warnings.append(
            f"{n_silent_errored} record(s) have a non-finite fitness "
            "(inf/-inf/nan) but an EMPTY `error` field -- this logger "
            "convention doesn't always populate `error` on failure. These "
            "are still counted/dropped as invalid (via non-finite fitness), "
            "but the `error` field alone would have undercounted them."
        )

    n_missing_fields = 0
    for r in records:
        missing = [f for f in REQUIRED_FIELDS if not hasattr(r, f)]
        n_missing_fields += len(missing)

    valid_fitness = np.array(
        [r.fitness for r in records if not r.is_invalid],
        dtype=float,
    )
    if valid_fitness.size:
        fitness_min = float(valid_fitness.min())
        fitness_max = float(valid_fitness.max())
        fitness_mean = float(valid_fitness.mean())
        fitness_skew = _skew(valid_fitness)
    else:
        fitness_min = fitness_max = fitness_mean = fitness_skew = float("nan")
        warnings.append("No non-error records with finite fitness values.")

    if valid_fitness.size and (fitness_min < -1e-6 or fitness_max > 1.0 + 1e-6):
        warnings.append(
            f"fitness range [{fitness_min:.4g}, {fitness_max:.4g}] falls "
            "outside the expected [0, 1] AOCC-style range -- verify the "
            "scale/direction convention for this file before pooling it "
            "with others."
        )

    frac_configspace_populated = sum(bool(r.configspace) for r in records) / n
    frac_operator_populated = sum(bool(r.operator) for r in records) / n
    frac_initial_population = sum(not r.parent_ids for r in records) / n

    generation_fitness_spearman = _generation_fitness_trend(records)
    if np.isfinite(generation_fitness_spearman) and generation_fitness_spearman < -0.3:
        warnings.append(
            "fitness trends DOWN with generation (Spearman "
            f"rho={generation_fitness_spearman:.2f}) -- if this run is "
            "supposed to be higher-is-better AOCC, the convention may be "
            "inverted relative to other files. Check manually."
        )

    return SchemaReport(
        file=label,
        n_records=n,
        n_errored=n_errored,
        error_fraction=error_fraction,
        n_missing_fields=n_missing_fields,
        fitness_min=fitness_min,
        fitness_max=fitness_max,
        fitness_mean=fitness_mean,
        fitness_skew=fitness_skew,
        frac_configspace_populated=frac_configspace_populated,
        frac_operator_populated=frac_operator_populated,
        frac_initial_population=frac_initial_population,
        generation_fitness_spearman=generation_fitness_spearman,
        warnings=warnings,
    )


def _skew(arr: np.ndarray) -> float:
    if arr.size < 3:
        return float("nan")
    from scipy import stats as scipy_stats

    return float(scipy_stats.skew(arr))


def _generation_fitness_trend(records: list[BladeRecord]) -> float:
    """Spearman rho between generation and fitness, as a direction sanity check."""
    valid = [(r.generation, r.fitness) for r in records if not r.is_invalid]
    if len(valid) < 5:
        return float("nan")
    from scipy import stats as scipy_stats

    gens, fits = zip(*valid)
    if len(set(gens)) < 2:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(gens, fits)
    return float(rho)
