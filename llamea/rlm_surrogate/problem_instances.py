"""Reconstructs the exact BBOB / MA-BBOB problem instance behind one
``metadata.aucs[i]`` entry and computes a cheap Latin Hypercube "fingerprint"
of it, so a training example can carry real problem-side signal -- not just
an aggregate ``fitness`` blind to which landscape produced it.

Background: each BladeRecord's ``fitness`` is the *mean* AOCC over many
problem instances, and ``metadata.aucs`` holds the per-instance breakdown.
Two real sources give the exact instance behind each ``aucs[i]``, confirmed
against real BLADE-results logs (both attached and read directly):

1. **``metadata.performance_data``** (BBOB only, present on every non-errored
   BBOB record seen): a list aligned 1:1 with ``aucs``, each entry
   self-describing -- ``{"fid": 1, "iid": 1, "dim": 10, "auc": 0.85...}``
   with ``performance_data[i]["auc"] == aucs[i]`` exactly. No external file
   needed.
2. **The sibling ``experimentlog.jsonl``** (MA_BBOB always, and the fallback
   for any BBOB record without ``performance_data``): one line per run,
   keyed by ``log_dir`` (the run folder name), carrying
   ``problem.training_instances`` -- for BBOB a literal list of ``[fid,
   iid]`` pairs, for MA_BBOB a string like ``"range(0, 10)"`` -- aligned 1:1
   with that run's ``aucs``, plus ``problem.dims`` (a single-element list in
   every run seen). Verified position-by-position identical to
   ``performance_data`` on a record from the same run.

``resolve_instances_for_record`` tries (1) then (2) and returns ``None`` if
neither works -- callers must skip that record rather than guess.

Requires the ``ioh`` extra (``uv sync --group rlm-surrogate``) only for
``reconstruct_problem``/``lhs_fingerprint``; imported lazily there so the
rest of the pipeline (including instance resolution) works without it.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

MA_BBOB_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "ma_bbob"

# The raw `problem.name` values seen in experimentlog.jsonl -- distinct from
# `data_pipeline.classify_problem`'s folder-derived `problem_id` grouping
# ("BBOB"/"MA-BBOB", hyphen). Don't conflate the two.
_BBOB_PROBLEM_NAME = "BBOB"
_MA_BBOB_PROBLEM_NAME = "MA_BBOB"


@dataclass(frozen=True)
class ProblemInstance:
    """One concrete, reconstructable problem instance."""

    kind: Literal["bbob", "ma_bbob"]
    dim: int
    fid_or_idx: int  # BBOB: function id 1..24. MA-BBOB: row idx into the CSVs.
    iid: int = 1  # BBOB only; unused for ma_bbob.


def instances_from_performance_data(
    performance_data: list[dict[str, Any]],
) -> list[ProblemInstance]:
    """Reads instances directly from a BBOB record's own
    ``metadata.performance_data`` -- no external file needed."""
    return [
        ProblemInstance(
            kind="bbob", dim=int(e["dim"]), fid_or_idx=int(e["fid"]), iid=int(e["iid"])
        )
        for e in performance_data
    ]


def _parse_range_string(s: str) -> list[int] | None:
    """Parses a logged Python ``range(a, b)`` repr into ``list(range(a, b))``
    without ``eval``. Returns ``None`` if it doesn't match that shape."""
    m = re.fullmatch(r"range\((\d+),\s*(\d+)\)", s.strip())
    if not m:
        return None
    return list(range(int(m.group(1)), int(m.group(2))))


def parse_training_instances(
    problem_spec: dict[str, Any],
) -> list[ProblemInstance] | None:
    """Reads the ordered instance list from an ``experimentlog.jsonl`` entry's
    ``problem`` block (``problem.name`` + ``problem.dims`` +
    ``problem.training_instances``). Returns ``None`` -- caller skips, never
    guesses -- when ``dims`` isn't a single value or the shape/name isn't
    recognized."""
    dims = problem_spec.get("dims") or []
    if len(dims) != 1:
        return None
    dim = int(dims[0])
    name = problem_spec.get("name")
    training_instances = problem_spec.get("training_instances")

    if name == _BBOB_PROBLEM_NAME:
        if not isinstance(training_instances, list):
            return None
        try:
            return [
                ProblemInstance(kind="bbob", dim=dim, fid_or_idx=int(fid), iid=int(iid))
                for fid, iid in training_instances
            ]
        except (TypeError, ValueError):
            return None

    if name == _MA_BBOB_PROBLEM_NAME:
        if isinstance(training_instances, str):
            idxs = _parse_range_string(training_instances)
        elif isinstance(training_instances, list):
            idxs = [int(i) for i in training_instances]
        else:
            idxs = None
        if idxs is None:
            return None
        return [
            ProblemInstance(kind="ma_bbob", dim=dim, fid_or_idx=idx) for idx in idxs
        ]

    return None


def load_experiment_instances(
    experimentlog_path: str | Path,
) -> dict[str, list[ProblemInstance]]:
    """Maps ``log_dir`` (run folder name) -> ordered instance list, from one
    ``experimentlog.jsonl`` file's ``problem`` blocks. Missing/malformed
    lines and entries whose `problem` block can't be parsed are skipped
    (not fatal -- other runs in the same file may still resolve)."""
    result: dict[str, list[ProblemInstance]] = {}
    path = Path(experimentlog_path)
    if not path.exists():
        return result
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            log_dir = d.get("log_dir")
            if not log_dir or log_dir in result:
                continue
            instances = parse_training_instances(d.get("problem", {}))
            if instances is not None:
                result[log_dir] = instances
    return result


def resolve_instances_for_record(
    record: Any, experiment_instance_cache: dict[str, dict[str, list[ProblemInstance]]]
) -> list[ProblemInstance] | None:
    """Resolves the ordered instance list behind one BladeRecord's
    ``metadata.aucs``: ``metadata.performance_data`` first, else the
    sibling ``experimentlog.jsonl`` two directories up from
    ``record.source_file`` (``<experiment_folder>/run-*/log.jsonl`` ->
    ``<experiment_folder>/experimentlog.jsonl``), looked up by the run
    folder's name. ``experiment_instance_cache`` is keyed by experiment
    folder path so each ``experimentlog.jsonl`` is read once per call site.
    Returns ``None`` if neither source resolves -- caller must skip."""
    metadata = record.metadata or {}
    if metadata.get("performance_data"):
        return instances_from_performance_data(metadata["performance_data"])

    source_file = Path(record.source_file)
    run_dir = source_file.parent
    experiment_dir = run_dir.parent
    key = str(experiment_dir)
    if key not in experiment_instance_cache:
        experiment_instance_cache[key] = load_experiment_instances(
            experiment_dir / "experimentlog.jsonl"
        )
    return experiment_instance_cache[key].get(run_dir.name)


@functools.lru_cache(maxsize=1)
def _load_ma_bbob_tables():
    import pandas as pd

    weights = pd.read_csv(MA_BBOB_DATA_DIR / "weights.csv", index_col=0)
    iids = pd.read_csv(MA_BBOB_DATA_DIR / "iids.csv", index_col=0)
    opt_locs = pd.read_csv(MA_BBOB_DATA_DIR / "opt_locs.csv", index_col=0)
    return weights, iids, opt_locs


def reconstruct_problem(instance: ProblemInstance) -> Any:
    """Rebuilds the ``ioh`` problem instance ``instance`` refers to. Lazily
    imports ``ioh``."""
    import ioh

    if instance.kind == "bbob":
        return ioh.get_problem(instance.fid_or_idx, instance.iid, instance.dim)
    if instance.kind == "ma_bbob":
        weights, iids_table, opt_locs = _load_ma_bbob_tables()
        idx = instance.fid_or_idx
        dim = instance.dim
        f_new = ioh.problem.ManyAffine(
            xopt=np.array(opt_locs.iloc[idx])[:dim],
            weights=np.array(weights.iloc[idx]),
            instances=np.array(iids_table.iloc[idx], dtype=int),
            n_variables=dim,
        )
        f_new.set_id(100)
        f_new.set_instance(idx)
        return f_new
    raise ValueError(f"Unknown ProblemInstance.kind: {instance.kind!r}")


def lhs_fingerprint(problem: Any, dim: int, n_points: int = 20, seed: int = 0) -> str:
    """Draws an ``n_points``-sample Latin Hypercube design over ``problem``'s
    domain, evaluates it, and formats it as compact text.

    Uses the same ``seed`` (hence identical probe coordinates) across every
    instance of a given ``dim``, so fingerprint differences reflect the
    function's shape, not where it was sampled -- makes fingerprints more
    directly comparable across landscapes."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n_points)
    lb = np.asarray(problem.bounds.lb, dtype=float)
    ub = np.asarray(problem.bounds.ub, dtype=float)
    xs = qmc.scale(unit, lb, ub)

    rows = []
    for x in xs:
        fx = float(problem(x))
        coords = ",".join(f"{v:.4g}" for v in x)
        rows.append(f"({coords})->{fx:.4g}")
    return f"LHS[{n_points}pts,dim={dim}]: " + "; ".join(rows)


def compute_problem_feature_text(
    instance: ProblemInstance, *, n_points: int = 20, seed: int = 0
) -> str:
    """Reconstructs ``instance`` and returns its LHS fingerprint text.
    Raises on failure (unknown ``ioh``/data issues) -- callers should catch,
    skip, and count/warn per the pipeline's "surface anomalies, never
    silently guess" convention."""
    problem = reconstruct_problem(instance)
    return lhs_fingerprint(problem, instance.dim, n_points=n_points, seed=seed)
