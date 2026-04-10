import os
import traceback
import numpy as np
import pandas as pd
import xgboost as xgb

from pflacco.sampling import create_initial_sample
from pflacco.classical_ela_features import (
    calculate_ela_meta,
    calculate_ela_distribution,
    calculate_nbc,
    calculate_dispersion,
    calculate_pca,
    calculate_information_content,
)
from pflacco.deep_ela import load_large_50d_v1


# -----------------------------
# Helpers
# -----------------------------
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    has_group = False
    if "group" in data.columns:
        group = data["group"].copy()
        data = data.drop("group", axis=1)
        has_group = True

    data = data.dropna(axis=1, how="any")
    data = data[data.columns.drop(list(data.filter(regex="costs_runtime")))]

    if has_group:
        data["group"] = group
    return data


def safe_scale_xy(X: pd.DataFrame, y: pd.Series):
    """
    Robust min-max scaling for X and y.
    Avoids undefined variables in constant-output cases.
    """
    X = X.copy()
    y = y.copy()

    # Scale X column-wise
    X_min = X.min()
    X_max = X.max()
    X_range = (X_max - X_min).replace(0, 1.0)
    X_scaled = (X - X_min) / X_range

    # Scale y safely
    y_min = float(y.min())
    y_max = float(y.max())
    if np.isclose(y_max, y_min):
        y_scaled = pd.Series(np.zeros(len(y)), index=y.index)
    else:
        y_scaled = (y - y_min) / (y_max - y_min)

    return X_scaled, y_scaled


def compute_classical_ela(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    X_scaled, y_scaled = safe_scale_xy(X, y)

    ela_meta = calculate_ela_meta(X_scaled, y_scaled)
    ela_distr = calculate_ela_distribution(X_scaled, y_scaled)
    nbc = calculate_nbc(X_scaled, y_scaled)
    disp = calculate_dispersion(X_scaled, y_scaled)
    pca = calculate_pca(X_scaled, y_scaled)
    ic = calculate_information_content(X_scaled, y_scaled)

    merged = {**ela_meta, **ela_distr, **nbc, **disp, **pca, **ic}
    merged_df = pd.DataFrame([{k: v for k, v in merged.items()}])
    merged_df = preprocess_data(merged_df)

    return merged_df.iloc[0]


def compute_deep_ela(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    out = model(X, y, include_costs=False)
    out_df = pd.DataFrame([out])
    out_df = preprocess_data(out_df)
    return out_df.iloc[0]


def build_feature_row(model, f, dim=5, n=250, seed=0):
    X = create_initial_sample(
        dim,
        n=n * dim,
        lower_bound=-5,
        upper_bound=5,
        seed=seed,
    )
    y = X.apply(f, axis=1)

    # Avoid log-related issues in downstream feature code if needed
    y = pd.Series(y).copy()
    y[y == 0] = 0.1 ** 100

    deep_row = compute_deep_ela(model, X, y)
    classical_row = compute_classical_ela(X, y)

    combined = pd.concat([deep_row, classical_row], axis=0)
    return X, y, deep_row, classical_row, combined


def try_xgb_prediction(model_path: str, feature_row: pd.Series):
    if not os.path.exists(model_path):
        print(f"[SKIP] XGBoost model not found: {model_path}")
        return None

    model = xgb.XGBClassifier(objective="binary:logistic")
    model.load_model(model_path)

    # Best effort: align columns if booster has feature names
    booster = model.get_booster()
    booster_features = booster.feature_names

    X_input = pd.DataFrame([feature_row])

    if booster_features is not None:
        missing = [c for c in booster_features if c not in X_input.columns]
        extra = [c for c in X_input.columns if c not in booster_features]

        print(f"Model expects {len(booster_features)} features")
        print(f"Input provides {X_input.shape[1]} features")
        print(f"Missing columns: {len(missing)}")
        if missing:
            print("First missing:", missing[:10])
        print(f"Extra columns: {len(extra)}")
        if extra:
            print("First extra:", extra[:10])

        # Fill missing with zero, drop extra, and reorder
        for c in missing:
            X_input[c] = 0.0
        X_input = X_input[booster_features]

    probs = model.predict_proba(X_input)[0]
    print("XGBoost prediction OK")
    print("predict_proba =", probs)
    return probs


# -----------------------------
# Example functions
# -----------------------------
def sphere(x):
    x = np.asarray(x)
    return float(np.sum(x ** 2))


def rastrigin(x):
    x = np.asarray(x)
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def separable_abs_sin(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x) + 0.1 * np.sin(5 * x)))


def mildly_nonseparable(x):
    x = np.asarray(x)
    return float(np.sum(x**2) + 0.2 * np.prod(np.tanh(x)))


def constant_function(x):
    return 1.0


TEST_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "separable_abs_sin": separable_abs_sin,
    "mildly_nonseparable": mildly_nonseparable,
    "constant_function": constant_function,
}


# -----------------------------
# Main smoke test
# -----------------------------
def main():
    print("Loading Deep-ELA model...")
    model_50 = load_large_50d_v1()
    print("Deep-ELA model loaded.\n")

    dims_to_test = [2, 5, 10]
    model_path = "new_models/model_Groups_GlobalLocal_50d+ela.json"  # change if needed

    for fname, f in TEST_FUNCTIONS.items():
        print("=" * 80)
        print(f"Testing function: {fname}")

        for dim in dims_to_test:
            print(f"\n--- dim={dim} ---")
            try:
                X, y, deep_row, classical_row, combined = build_feature_row(
                    model=model_50,
                    f=f,
                    dim=dim,
                    n=250,
                    seed=42,
                )

                print(f"Sample shape: X={X.shape}, y={y.shape}")
                print(f"Deep-ELA features: {len(deep_row)}")
                print(f"Classical ELA features: {len(classical_row)}")
                print(f"Combined features: {len(combined)}")

                print("First 8 Deep-ELA features:")
                print(deep_row.head(8))

                print("First 8 Classical ELA features:")
                print(classical_row.head(8))

                # Optional XGBoost compatibility test
                try_xgb_prediction(model_path, combined)

            except Exception as e:
                print(f"[FAIL] {fname} at dim={dim}: {e}")
                traceback.print_exc()

        print()

    print("Done.")


if __name__ == "__main__":
    main()