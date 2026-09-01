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


def _lhs_sample(
    problem: Any, dim: int, n_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draws an ``n_points``-sample Latin Hypercube design over ``problem``'s
    domain and evaluates it, returning ``(xs, ys)``.

    Uses the same ``seed`` (hence identical probe coordinates) across every
    instance of a given ``dim``, so downstream features differ because of
    the function's shape, not where it was sampled -- makes them more
    directly comparable across landscapes. Shared by ``lhs_fingerprint``
    (Tier 1: raw text) and ``lhs_summary_stats`` (Tier B: computed stats)
    so both spend the same evaluation budget on the same points."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n_points)
    lb = np.asarray(problem.bounds.lb, dtype=float)
    ub = np.asarray(problem.bounds.ub, dtype=float)
    xs = qmc.scale(unit, lb, ub)
    ys = np.array([float(problem(x)) for x in xs])
    return xs, ys


def lhs_fingerprint(problem: Any, dim: int, n_points: int = 20, seed: int = 0) -> str:
    """Draws an ``n_points``-sample Latin Hypercube design over ``problem``'s
    domain, evaluates it, and formats it as compact raw ``(x)->f(x)`` text."""
    xs, ys = _lhs_sample(problem, dim, n_points, seed)
    rows = []
    for x, fx in zip(xs, ys):
        coords = ",".join(f"{v:.4g}" for v in x)
        rows.append(f"({coords})->{fx:.4g}")
    return f"LHS[{n_points}pts,dim={dim}]: " + "; ".join(rows)


def lhs_summary_stats(problem: Any, dim: int, n_points: int = 20, seed: int = 0) -> str:
    """Tier B: computes a handful of cheap landscape-summary statistics from
    the *same* LHS sample ``lhs_fingerprint`` would draw, instead of dumping
    the raw ``(x)->f(x)`` pairs as text -- same evaluation budget, more
    directly usable signal (the model doesn't have to infer distributional
    shape from raw numbers itself).

    Statistics: y-distribution mean/std/skewness/kurtosis and coefficient of
    variation, plus a cheap diagonal-quadratic meta-model (``y ~ c0 + b*x +
    sum(a_i * x_i^2)``) fit via least squares -- its R^2 is a nonlinearity/
    multimodality proxy, and the ratio of its largest to smallest |a_i| is a
    rough conditioning-number estimate. This is a lightweight, ELA-inspired
    approximation, not a substitute for a proper ELA feature set (e.g.
    ``pflacco``) -- meaningful mainly when ``n_points`` comfortably exceeds
    ``2*dim + 1`` (the diagonal model's parameter count); below that the
    least-squares fit is under-determined and the quad/cond numbers are
    noisier."""
    from scipy import stats as scipy_stats

    xs, ys = _lhs_sample(problem, dim, n_points, seed)

    y_mean = float(np.mean(ys))
    y_std = float(np.std(ys))
    y_skew = float(scipy_stats.skew(ys)) if n_points >= 3 else float("nan")
    y_kurtosis = float(scipy_stats.kurtosis(ys)) if n_points >= 4 else float("nan")
    cv = y_std / abs(y_mean) if abs(y_mean) > 1e-12 else float("nan")

    design = np.column_stack([np.ones(n_points), xs, xs**2])
    coeffs, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
    y_fit = design @ coeffs
    ss_res = float(np.sum((ys - y_fit) ** 2))
    ss_tot = float(np.sum((ys - y_mean) ** 2))
    quad_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    quad_coeffs = np.abs(coeffs[1 + dim :])
    quad_coeffs = quad_coeffs[quad_coeffs > 1e-12]
    cond_est = (
        float(quad_coeffs.max() / quad_coeffs.min())
        if len(quad_coeffs) > 0
        else float("nan")
    )

    return (
        f"STATS[{n_points}pts,dim={dim}]: y_mean={y_mean:.4g}; y_std={y_std:.4g}; "
        f"y_skew={y_skew:.4g}; y_kurtosis={y_kurtosis:.4g}; cv={cv:.4g}; "
        f"quad_r2={quad_r2:.4g}; cond_est={cond_est:.4g}"
    )


def describe_ma_bbob_composition(idx: int, weight_threshold: float = 0.01) -> str:
    """Tier A for MA-BBOB: MA-BBOB instances are affine combinations of a
    handful of the 24 canonical BBOB functions (see
    ``benchmarks/ma_bbob/weights.csv``, one row per instance, one column per
    base function, mostly zero). Reads row ``idx``, keeps components whose
    weight exceeds ``weight_threshold``, and describes each via
    ``bbob_properties.describe_bbob_function`` -- so an MA-BBOB row gets the
    same free, zero-extra-evaluation problem info a BBOB row does."""
    from . import bbob_properties

    weights, _, _ = _load_ma_bbob_tables()
    row = weights.iloc[idx]
    # Columns are 0-indexed by position (column "0" is the weight on BBOB
    # f1, column "23" on f24) -- same convention `reconstruct_problem`
    # relies on when passing this row straight through to
    # `ioh.problem.ManyAffine(weights=...)`. Off-by-one here would silently
    # mislabel every MA-BBOB composition.
    components = sorted(
        (
            (int(col) + 1, float(w))
            for col, w in row.items()
            if abs(w) > weight_threshold
        ),
        key=lambda t: -t[1],
    )
    parts = [
        f"{w * 100:.1f}% {bbob_properties.describe_bbob_function(fid)}"
        for fid, w in components
    ]
    return "MA-BBOB combination: " + " + ".join(parts)


def compute_meta_feature_text(instance: ProblemInstance) -> str:
    """Tier A: static, free (no function evaluations, no ``ioh`` needed)
    problem-meta-feature text -- dimensionality plus known BBOB function
    properties (or, for MA-BBOB, its base-function composition). Callers
    (``compute_problem_feature_text``, ``data_pipeline.py``) own the
    surrounding "# Problem" header -- this returns body text only."""
    if instance.kind == "bbob":
        from . import bbob_properties

        return (
            f"family: BBOB; dim: {instance.dim}; "
            f"{bbob_properties.describe_bbob_function(instance.fid_or_idx)}"
        )
    if instance.kind == "ma_bbob":
        return (
            f"family: MA-BBOB; dim: {instance.dim}; "
            f"{describe_ma_bbob_composition(instance.fid_or_idx)}"
        )
    raise ValueError(f"Unknown ProblemInstance.kind: {instance.kind!r}")


FeatureMode = Literal["lhs", "lhs_stats", "meta", "meta+lhs", "meta+lhs_stats"]


def compute_problem_feature_text(
    instance: ProblemInstance,
    *,
    n_points: int = 20,
    seed: int = 0,
    mode: FeatureMode = "lhs",
) -> str:
    """Builds the "# Problem" text appended to a training example's ``x``.

    ``mode`` selects which tier(s) to include:
      - ``"lhs"`` (default, unchanged from the original shipped behavior):
        raw Latin Hypercube ``(x)->f(x)`` sample text. Requires ``ioh``.
      - ``"lhs_stats"``: Tier B computed summary statistics from the same
        LHS sample, instead of the raw text. Requires ``ioh``.
      - ``"meta"``: Tier A static properties only (dimensionality + known
        BBOB/MA-BBOB structure). No function evaluations, no ``ioh`` needed.
      - ``"meta+lhs"`` / ``"meta+lhs_stats"``: both, concatenated.

    Raises on failure (unknown ``ioh``/data issues, or ``mode="meta"`` for a
    function id/idx this module doesn't know) -- callers should catch, skip,
    and count/warn per the pipeline's "surface anomalies, never silently
    guess" convention.
    """
    parts = []
    if mode.startswith("meta"):
        parts.append(compute_meta_feature_text(instance))
    if mode in ("lhs", "meta+lhs"):
        problem = reconstruct_problem(instance)
        parts.append(
            lhs_fingerprint(problem, instance.dim, n_points=n_points, seed=seed)
        )
    elif mode in ("lhs_stats", "meta+lhs_stats"):
        problem = reconstruct_problem(instance)
        parts.append(
            lhs_summary_stats(problem, instance.dim, n_points=n_points, seed=seed)
        )
    elif mode != "meta":
        raise ValueError(f"Unknown feature mode: {mode!r}")
    return "\n".join(parts)
