"""Static, free (zero extra function evaluations) knowledge about the 24
canonical BBOB functions -- their COCO documentation grouping, separability,
unimodality, and conditioning -- used to build "Tier A" problem-meta-feature
text in ``problem_instances.py``.

The 5-group split (separable / low-or-moderate conditioning / high
conditioning unimodal / multi-modal with adequate global structure /
multi-modal with weak global structure) is the standard COCO/BBOB grouping
(Hansen et al., "Real-Parameter Black-Box Optimization Benchmarking":
Noiseless Functions Definitions) and is safe to treat as authoritative.
The per-function ``separable``/``unimodal``/``conditioning`` fields below are
best-effort, drawn from that same documentation and widely-cited summaries
of it -- double-check them against the official function definitions before
relying on them for anything publication-grade; a subtly wrong tag here
would feed misleading "ground truth" problem info into training data, so
this table should be treated as a first draft, not gospel.

MA-BBOB instances are affine combinations of a handful of these 24 base
functions (see ``benchmarks/ma_bbob/weights.csv``, read by
``problem_instances.describe_ma_bbob_composition``) -- reusing this same
table lets an MA-BBOB row be described just as informatively as a BBOB row,
for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Group = Literal[1, 2, 3, 4, 5]

GROUP_NAMES: dict[Group, str] = {
    1: "separable",
    2: "low-or-moderate conditioning",
    3: "high conditioning, unimodal",
    4: "multi-modal, adequate global structure",
    5: "multi-modal, weak global structure",
}


@dataclass(frozen=True)
class BBOBFunctionInfo:
    fid: int
    name: str
    group: Group
    separable: bool
    unimodal: bool
    conditioning: Literal["low", "moderate", "high", "varies"]


BBOB_FUNCTIONS: dict[int, BBOBFunctionInfo] = {
    info.fid: info
    for info in [
        BBOBFunctionInfo(1, "Sphere", 1, True, True, "low"),
        BBOBFunctionInfo(2, "Ellipsoidal (separable)", 1, True, True, "high"),
        BBOBFunctionInfo(3, "Rastrigin (separable)", 1, True, False, "low"),
        BBOBFunctionInfo(4, "Bueche-Rastrigin", 1, True, False, "low"),
        BBOBFunctionInfo(5, "Linear Slope", 1, True, True, "low"),
        BBOBFunctionInfo(6, "Attractive Sector", 2, False, True, "high"),
        BBOBFunctionInfo(7, "Step Ellipsoidal", 2, False, True, "high"),
        BBOBFunctionInfo(8, "Rosenbrock (original)", 2, False, True, "moderate"),
        BBOBFunctionInfo(9, "Rosenbrock (rotated)", 2, False, True, "moderate"),
        BBOBFunctionInfo(10, "Ellipsoidal (rotated)", 3, False, True, "high"),
        BBOBFunctionInfo(11, "Discus", 3, False, True, "high"),
        BBOBFunctionInfo(12, "Bent Cigar", 3, False, True, "high"),
        BBOBFunctionInfo(13, "Sharp Ridge", 3, False, True, "high"),
        BBOBFunctionInfo(14, "Different Powers", 3, False, True, "moderate"),
        BBOBFunctionInfo(15, "Rastrigin (rotated)", 4, False, False, "moderate"),
        BBOBFunctionInfo(16, "Weierstrass", 4, False, False, "moderate"),
        BBOBFunctionInfo(17, "Schaffers F7", 4, False, False, "low"),
        BBOBFunctionInfo(
            18, "Schaffers F7, moderately ill-conditioned", 4, False, False, "moderate"
        ),
        BBOBFunctionInfo(
            19, "Composite Griewank-Rosenbrock F8F2", 4, False, False, "moderate"
        ),
        BBOBFunctionInfo(20, "Schwefel", 5, False, False, "low"),
        BBOBFunctionInfo(
            21, "Gallagher's Gaussian 101-me Peaks", 5, False, False, "varies"
        ),
        BBOBFunctionInfo(
            22, "Gallagher's Gaussian 21-hi Peaks", 5, False, False, "varies"
        ),
        BBOBFunctionInfo(23, "Katsuura", 5, False, False, "moderate"),
        BBOBFunctionInfo(24, "Lunacek bi-Rastrigin", 5, False, False, "moderate"),
    ]
}


def describe_bbob_function(fid: int) -> str:
    """Renders ``fid``'s known properties as a compact text fragment.
    Raises ``KeyError`` for an fid outside 1..24 -- callers should treat
    that as "can't describe," not silently skip."""
    info = BBOB_FUNCTIONS[fid]
    modality = "unimodal" if info.unimodal else "multi-modal"
    return (
        f"f{info.fid} {info.name} (group {info.group}: {GROUP_NAMES[info.group]}; "
        f"separable={info.separable}; {modality}; conditioning={info.conditioning})"
    )
