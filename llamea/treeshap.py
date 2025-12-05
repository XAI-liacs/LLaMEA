"""Light-weight wrapper around XGBoost's TreeSHAP implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb


@dataclass
class TreeExplainer:
    """Compute SHAP values for tree-based models using XGBoost."""

    booster: xgb.Booster

    def shap_values(self, matrix: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(matrix)
        contributions = self.booster.predict(dmatrix, pred_contribs=True)
        # The last column contains the expected value baseline.
        if contributions.ndim == 2 and contributions.shape[1] > 1:
            return contributions[:, :-1]
        return contributions


__all__ = ["TreeExplainer"]
