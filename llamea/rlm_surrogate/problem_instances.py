"""Reconstructs the exact BBOB / MA-BBOB problem instance behind one
``metadata.aucs[i]`` entry and computes a cheap Latin Hypercube "fingerprint"
of it, so a training example can carry real problem-side signal -- not just
an aggregate ``fitness`` blind to which landscape produced it.

Background: each BladeRecord's ``fitness`` is the *mean* AOCC over many
problem instances (see ``benchmarks/ma_bbob/run_mabbob.py`` and
``examples/black-box-optimization.py``), and ``metadata.aucs`` holds the
per-instance breakdown in a fixed loop order. ``InstanceSweepConfig``
describes that loop order so index ``i`` can be decoded back to which
instance produced ``aucs[i]``.

IMPORTANT CAVEAT: the two shipped presets (``BBOB_DEFAULT``,
``MA_BBOB_DEFAULT``) mirror the two evaluation scripts in this repo, but
have NOT been verified against real ``BLADE-results`` logs (the data
wasn't reachable while this module was written). If a real file's
``len(aucs)`` doesn't match either preset's ``expected_length``,
``match_sweep_config`` returns ``None`` and the caller must skip that
record rather than guess -- see ``data_pipeline.explode_aucs_with_problem_features``.
Build a custom ``InstanceSweepConfig`` (optionally from YAML via
``--sweep-config``) once the real sweep is known.

Requires the ``ioh`` extra (``uv sync --group rlm-surrogate``). Imported
lazily so the rest of the pipeline works without it installed.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

MA_BBOB_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "ma_bbob"


@dataclass
class InstanceSweepConfig:
    """Describes the fixed nested-loop order used to build one run's
    ``metadata.aucs``, so index ``i`` can be decoded back to a specific
    problem instance.

    Matches ``for dim in dims: for fid_or_idx in fids_or_idxs: for iid in
    iids: for rep in range(reps):`` (outermost to innermost) -- the order
    both reference evaluation scripts in this repo use. ``iids``/``reps``
    default to length-1 for MA-BBOB, which doesn't loop over them.
    """

    kind: Literal["bbob", "ma_bbob"]
    dims: list[int]
    fids_or_idxs: list[int]  # BBOB: function ids 1..24. MA-BBOB: table row idx.
    iids: list[int] = dataclasses.field(default_factory=lambda: [1])
    reps: int = 1

    @property
    def expected_length(self) -> int:
        return len(self.dims) * len(self.fids_or_idxs) * len(self.iids) * self.reps

    def decode(self, i: int) -> dict[str, int]:
        """Returns ``{"dim", "fid_or_idx", "iid", "rep"}`` for ``aucs[i]``."""
        if not 0 <= i < self.expected_length:
            raise IndexError(
                f"{i} out of range for expected_length={self.expected_length}"
            )
        n_reps, n_iids, n_fids = self.reps, len(self.iids), len(self.fids_or_idxs)
        rep = i % n_reps
        i //= n_reps
        iid = self.iids[i % n_iids]
        i //= n_iids
        fid_or_idx = self.fids_or_idxs[i % n_fids]
        i //= n_fids
        dim = self.dims[i]
        return {"dim": dim, "fid_or_idx": fid_or_idx, "iid": iid, "rep": rep}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InstanceSweepConfig":
        import yaml

        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as fh:
            yaml.safe_dump(dataclasses.asdict(self), fh, sort_keys=False)


# Mirrors examples/black-box-optimization.py:
#   for dim in [5]: for fid in range(1,25): for iid in [1,2,3]: for rep in range(3):
BBOB_DEFAULT = InstanceSweepConfig(
    kind="bbob", dims=[5], fids_or_idxs=list(range(1, 25)), iids=[1, 2, 3], reps=3
)

# Mirrors benchmarks/ma_bbob/run_mabbob.py:
#   for dim in [2, 5]: for idx in range(100):
# (weights.csv/iids.csv/opt_locs.csv actually hold 1000 rows -- the script
# only used the first 100; widen fids_or_idxs if a run used more.)
MA_BBOB_DEFAULT = InstanceSweepConfig(
    kind="ma_bbob", dims=[2, 5], fids_or_idxs=list(range(100)), iids=[1], reps=1
)


def match_sweep_config(
    aucs_length: int, candidates: list[InstanceSweepConfig]
) -> InstanceSweepConfig | None:
    """Returns the first candidate whose ``expected_length`` matches
    ``aucs_length``, else ``None`` -- callers must skip on ``None``, never
    guess which sweep produced a mismatched-length ``aucs``."""
    for cfg in candidates:
        if cfg.expected_length == aucs_length:
            return cfg
    return None


@functools.lru_cache(maxsize=1)
def _load_ma_bbob_tables():
    import pandas as pd

    weights = pd.read_csv(MA_BBOB_DATA_DIR / "weights.csv", index_col=0)
    iids = pd.read_csv(MA_BBOB_DATA_DIR / "iids.csv", index_col=0)
    opt_locs = pd.read_csv(MA_BBOB_DATA_DIR / "opt_locs.csv", index_col=0)
    return weights, iids, opt_locs


def reconstruct_problem(cfg: InstanceSweepConfig, decoded: dict[str, int]) -> Any:
    """Rebuilds the ``ioh`` problem instance ``decoded`` (from
    ``InstanceSweepConfig.decode``) refers to. Lazily imports ``ioh``."""
    import ioh

    dim = decoded["dim"]
    if cfg.kind == "bbob":
        return ioh.get_problem(decoded["fid_or_idx"], decoded["iid"], dim)
    if cfg.kind == "ma_bbob":
        weights, iids_table, opt_locs = _load_ma_bbob_tables()
        idx = decoded["fid_or_idx"]
        f_new = ioh.problem.ManyAffine(
            xopt=np.array(opt_locs.iloc[idx])[:dim],
            weights=np.array(weights.iloc[idx]),
            instances=np.array(iids_table.iloc[idx], dtype=int),
            n_variables=dim,
        )
        f_new.set_id(100)
        f_new.set_instance(idx)
        return f_new
    raise ValueError(f"Unknown InstanceSweepConfig.kind: {cfg.kind!r}")


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
    cfg: InstanceSweepConfig, aucs_index: int, *, n_points: int = 20, seed: int = 0
) -> str:
    """Decodes ``aucs_index`` under ``cfg``, reconstructs the problem, and
    returns its LHS fingerprint text. Raises on failure (unknown
    ``ioh``/data issues) -- callers should catch, skip, and count/warn per
    the pipeline's "surface anomalies, never silently guess" convention."""
    decoded = cfg.decode(aucs_index)
    problem = reconstruct_problem(cfg, decoded)
    return lhs_fingerprint(problem, decoded["dim"], n_points=n_points, seed=seed)
