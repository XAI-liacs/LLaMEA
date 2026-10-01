import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


# Set this to the directory containing the exp_* folders.
EXPERIMENTS_DIR = Path("/local/bodasap/exp_test_2")

MODELS_DIR = Path("/local/bodasap/LLaMEA-ELA/models_fixed_labels")
DIMS = [2, 5, 10]


def load_model(feature):
    model_feature = feature.replace("NOT ", "")
    model_path = MODELS_DIR / f"model_Groups_{model_feature}_llamea_fixed_labels.json"
    model = xgb.XGBClassifier(objective="binary:logistic")
    model.load_model(model_path)
    return model


def calculate_score(metadata, feature, dim, model):
    deep_features = metadata[f"ela_deep_50d_{dim}D"]
    ela_key = "ela_features" if dim == 5 else f"ela_features_{dim}D"
    ela_features = metadata[ela_key]

    expected_features = model.get_booster().feature_names
    ela_names = [
        name
        for name in expected_features
        if name not in deep_features and name != "dim"
    ]

    input_row = {
        **deep_features,
        "dim": dim,
        **dict(zip(ela_names, ela_features)),
    }
    input_df = pd.DataFrame([input_row])[expected_features]
    score = float(model.predict_proba(input_df)[0][1])

    if feature.startswith("NOT "):
        score = 1.0 - score
    return score


def update_log(log_path):
    print(f"\nReading {log_path}")
    with log_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    print(f"Found {len(records)} records")

    features = []
    for record in records:
        fitness = record.get("fitness")
        if not isinstance(fitness, (int, float)) or not math.isfinite(fitness) or fitness <= 0:
            continue

        # Infer features from the first record with a valid positive fitness.
        for key in record["metadata"]:
            if key.startswith("score_") and key.endswith("_2D"):
                features.append(key[len("score_") : -len("_2D")])

        if features:
            break

    if not features:
        raise ValueError(f"No valid record with feature score keys found in {log_path}")

    models = {feature: load_model(feature) for feature in features}
    print(f"Loaded models for: {', '.join(models)}")

    updated = 0
    failed = 0
    for index, record in enumerate(records, start=1):
        metadata = record["metadata"]
        scores = []

        try:
            for dim in DIMS:
                for feature in features:
                    score_key = f"score_{feature}_{dim}D"
                    score = calculate_score(metadata, feature, dim, models[feature])
                    metadata[score_key] = score
                    scores.append(score)

            record["fitness"] = float(np.mean(scores))
            updated += 1
        except KeyError:
            # Failed evaluations may not contain all dimensions.
            record["fitness"] = float("-inf")
            failed += 1
            print(f"Record {index}/{len(records)} is incomplete; fitness set to -inf")
            continue

        if index % 25 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} records")

    output_path = log_path.with_name("log_updated_fixed_labels.jsonl")
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    print(f"Created {output_path}")
    print(f"Updated: {updated}, incomplete: {failed}")

print("runs")
for experiment_dir in sorted(EXPERIMENTS_DIR.glob("exp-*")):
    print(f"\nProcessing {experiment_dir}")
    log_path = experiment_dir / "log.jsonl"
    if log_path.exists():
        update_log(log_path)
