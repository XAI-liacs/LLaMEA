"""Lightweight hand-featured baseline, used in Step 3 to quantify how much
the RLM's "no feature engineering, raw code text" approach buys over a
cheap alternative (code length + AST/complexity features + a small
gradient-boosted regressor, via ``xgboost`` which is already an LLaMEA
dependency -- see ``llamea/ast_features.py`` and ``llamea/feature_guidance.py``
for the same pattern used elsewhere in this repo).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from llamea.ast_features import extract_ast_features

from .data_pipeline import RLMExample


def extract_code_features(code: str) -> dict[str, float]:
    """Code length proxies + AST/complexity features. Never raises: if the
    code fails to parse, falls back to length-only features."""
    lines = code.splitlines()
    features: dict[str, float] = {
        "char_count": float(len(code)),
        "line_count": float(len(lines)),
        "mean_line_length": float(np.mean([len(l) for l in lines])) if lines else 0.0,
    }
    try:
        features.update(extract_ast_features(code))
        features["parse_failed"] = 0.0
    except Exception:
        features["parse_failed"] = 1.0
    return features


def _extract_code_from_x(x: str) -> str:
    """``x`` may have description/configspace prepended (see
    ``data_pipeline.build_x``); the AST/complexity features only make sense
    on the actual code, so this strips back to the ``# Code`` section."""
    marker = "# Code\n"
    idx = x.rfind(marker)
    return x[idx + len(marker) :] if idx != -1 else x


class FeatureMatrixBuilder:
    """Builds an aligned numeric feature matrix across train/val/test splits.

    Column order is frozen on the first call (normally on the training set)
    so later splits reuse identical columns, filling any features absent in
    that split with 0.0.
    """

    def __init__(self) -> None:
        self.feature_names: list[str] | None = None

    def fit_transform(self, examples: Sequence[RLMExample]) -> np.ndarray:
        per_example = [
            extract_code_features(_extract_code_from_x(e.x)) for e in examples
        ]
        self.feature_names = sorted({k for d in per_example for k in d})
        return self._to_matrix(per_example)

    def transform(self, examples: Sequence[RLMExample]) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Call fit_transform before transform.")
        per_example = [
            extract_code_features(_extract_code_from_x(e.x)) for e in examples
        ]
        return self._to_matrix(per_example)

    def _to_matrix(self, per_example: list[dict[str, float]]) -> np.ndarray:
        matrix = np.array(
            [[d.get(name, 0.0) for name in self.feature_names] for d in per_example],
            dtype=float,
        )
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


class XGBFeatureBaseline:
    """Thin wrapper around xgboost for the hand-featured baseline regressor."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBFeatureBaseline":
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "eta": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "lambda": 1.0,
            "verbosity": 0,
            "seed": self.seed,
        }
        num_round = min(200, max(20, X.shape[0] * 5))
        self._booster = xgb.train(params, dtrain, num_boost_round=num_round)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        if self._booster is None:
            raise RuntimeError("Call fit before predict.")
        return self._booster.predict(xgb.DMatrix(X))
