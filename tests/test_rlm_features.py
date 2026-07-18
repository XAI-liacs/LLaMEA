import numpy as np

from llamea.rlm_surrogate.data_pipeline import RLMExample
from llamea.rlm_surrogate.features import (
    FeatureMatrixBuilder,
    XGBFeatureBaseline,
    _extract_code_from_x,
    extract_code_features,
)

SIMPLE_CODE = "class A:\n    def __init__(self):\n        self.x = 1\n"


def test_extract_code_features_valid_code():
    features = extract_code_features(SIMPLE_CODE)
    assert features["char_count"] == len(SIMPLE_CODE)
    assert features["line_count"] == len(SIMPLE_CODE.splitlines())
    assert features["parse_failed"] == 0.0
    assert "mean_complexity" in features  # from extract_ast_features


def test_extract_code_features_invalid_code_falls_back():
    features = extract_code_features("def f(:\n  broken")
    assert features["parse_failed"] == 1.0
    assert features["char_count"] > 0


def test_extract_code_from_x_strips_prepended_context():
    x = "# Description\nsomething\n\n# Code\n" + SIMPLE_CODE
    assert _extract_code_from_x(x) == SIMPLE_CODE


def test_extract_code_from_x_no_marker_returns_input():
    assert _extract_code_from_x(SIMPLE_CODE) == SIMPLE_CODE


def test_feature_matrix_builder_consistent_columns_across_splits():
    train = [RLMExample("1", "r", 0, [], SIMPLE_CODE, 0.5, 0.5)]
    test = [RLMExample("2", "r", 0, [], "class B:\n    pass\n", 0.6, 0.6)]
    builder = FeatureMatrixBuilder()
    X_train = builder.fit_transform(train)
    X_test = builder.transform(test)
    assert X_train.shape[1] == X_test.shape[1] == len(builder.feature_names)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_xgb_feature_baseline_fits_and_predicts_with_correct_shape():
    rng = np.random.default_rng(0)
    codes = [f"class A{i}:\n    x = {i}\n" for i in range(20)]
    examples = [
        RLMExample(str(i), "r", 0, [], c, float(i) / 20, float(i) / 20)
        for i, c in enumerate(codes)
    ]
    builder = FeatureMatrixBuilder()
    X = builder.fit_transform(examples)
    y = np.array([e.fitness for e in examples])
    model = XGBFeatureBaseline(seed=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
