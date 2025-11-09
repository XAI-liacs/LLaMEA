"""Feature-guided mutation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .ast_features import extract_ast_features
from .solution import Solution


@dataclass(frozen=True)
class FeatureGuidance:
    """Description of the guidance extracted from the archive model."""

    feature_name: str
    action: str
    message: str


def _prepare_training_data(
    solutions: Iterable[Solution], minimization: bool
) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[None, None, None]:
    feature_dicts: list[dict[str, float]] = []
    targets: list[float] = []

    for solution in solutions:
        code = getattr(solution, "code", "") or ""
        if not code.strip():
            continue

        fitness = getattr(solution, "fitness", None)
        if fitness is None or not np.isfinite(fitness):
            continue

        target_value = float(fitness)
        if minimization:
            target_value = -target_value

        features = solution.get_metadata("ast_features")
        if features is None:
            try:
                features = extract_ast_features(code)
            except Exception:
                continue
            solution.add_metadata("ast_features", dict(features))

        feature_dicts.append(features)
        targets.append(target_value)

    if len(feature_dicts) < 3:
        return None, None, None

    feature_names = sorted({key for d in feature_dicts for key in d})
    if not feature_names:
        return None, None, None

    matrix = np.array(
        [
            [float(feature_dict.get(name, np.nan)) for name in feature_names]
            for feature_dict in feature_dicts
        ],
        dtype=float,
    )
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)

    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    matrix = matrix[valid_rows]
    targets_arr = np.array(targets, dtype=float)[valid_rows]

    if matrix.shape[0] < 3:
        return None, None, None

    col_means = np.nanmean(matrix, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_indices = np.where(np.isnan(matrix))
    if nan_indices[0].size:
        matrix[nan_indices] = np.take(col_means, nan_indices[1])

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    targets_arr = np.nan_to_num(targets_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if np.allclose(matrix, matrix[0]) or np.allclose(targets_arr, targets_arr[0]):
        return None, None, None

    return matrix, targets_arr, feature_names


def compute_feature_guidance(
    solutions: Sequence[Solution], minimization: bool
) -> Optional[FeatureGuidance]:
    """Return guidance extracted from an archive of evaluated solutions."""

    try:
        from treeshap import TreeExplainer
    except Exception:  # pragma: no cover - handled gracefully in production
        return None

    matrix, targets, feature_names = _prepare_training_data(solutions, minimization)
    if matrix is None or targets is None or feature_names is None:
        return None

    try:
        import xgboost as xgb
    except Exception:  # pragma: no cover - handled gracefully in production
        return None

    dtrain = xgb.DMatrix(matrix, label=targets)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "lambda": 1.0,
        "verbosity": 0,
        "seed": 0,
    }

    num_round = min(200, max(50, matrix.shape[0] * 10))

    try:
        booster = xgb.train(params, dtrain, num_boost_round=num_round)
    except Exception:
        return None

    explainer = TreeExplainer(booster)
    shap_values = np.array(explainer.shap_values(matrix))
    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    if shap_values.size == 0:
        return None

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    if not np.any(mean_abs > 0):
        return None

    best_idx = int(np.argmax(mean_abs))
    feature_column = matrix[:, best_idx]
    shap_column = shap_values[:, best_idx]

    valid = np.isfinite(feature_column) & np.isfinite(shap_column)
    if valid.sum() < 2:
        return None

    feature_column = feature_column[valid]
    shap_column = shap_column[valid]

    centered_feature = feature_column - feature_column.mean()
    variance = float(np.mean(centered_feature**2))
    if variance <= 1e-12:
        return None

    centered_shap = shap_column - shap_column.mean()
    covariance = float(np.mean(centered_feature * centered_shap))
    if abs(covariance) <= 1e-12:
        return None

    slope = covariance / variance
    action = "increase" if slope > 0 else "decrease"

    feature_label = feature_names[best_idx]
    readable_feature = feature_label.replace("_", " ").lower()
    message = (
        "Based on archive analysis, try to "
        f"{action} the {readable_feature} of the solution."
    )

    return FeatureGuidance(
        feature_name=feature_label,
        action=action,
        message=message,
    )


__all__ = ["FeatureGuidance", "compute_feature_guidance"]

