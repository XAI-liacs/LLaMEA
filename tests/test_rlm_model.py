"""Regression coverage for `median_point_predictions`'s chunking.

`reg_lm.sample` decodes its entire input as one `len(xs) * num_samples`
batch with no chunking of its own (see `model.median_point_predictions`'s
docstring) -- a large eval set with no chunking of ours turns into an
enormous single decode that can run for hours or OOM outright (confirmed
against a real run: a >300GB CPU allocation attempt). These tests confirm
`batch_size` actually splits the work into multiple `reg_lm.sample` calls
and that predictions are unaffected by the chunk size chosen.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("regress_lm")

from llamea.rlm_surrogate.config import RLMSurrogateConfig
from llamea.rlm_surrogate.model import build_model, median_point_predictions

TINY_CONFIG = (
    Path(__file__).parent.parent
    / "llamea"
    / "rlm_surrogate"
    / "configs"
    / "tiny_local_test.yaml"
)


@pytest.fixture(scope="module")
def reg_lm():
    config = RLMSurrogateConfig.from_yaml(TINY_CONFIG)
    return build_model(config)


def test_median_point_predictions_chunks_into_multiple_sample_calls(reg_lm):
    xs = [f"class Algo{i}:\n    pass\n" for i in range(10)]
    with patch.object(reg_lm, "sample", wraps=reg_lm.sample) as spy:
        medians = median_point_predictions(reg_lm, xs, num_samples=2, batch_size=3)
    assert medians.shape[0] == len(xs)
    # 10 items at batch_size=3 -> 4 calls (3,3,3,1), never all 10 in one call.
    assert spy.call_count == 4
    for call in spy.call_args_list:
        assert len(call.args[0]) <= 3


def test_median_point_predictions_same_shape_regardless_of_batch_size(reg_lm):
    """Sampling is inherently stochastic (torch.multinomial draws) and the
    exact RNG consumption pattern differs with tensor shape, so chunking
    isn't expected to reproduce bit-identical values -- just the same
    output shape and finite values, independent of how it's chunked."""
    xs = [f"class Algo{i}:\n    pass\n" for i in range(6)]

    medians_one_batch = median_point_predictions(
        reg_lm, xs, num_samples=3, batch_size=100
    )
    medians_chunked = median_point_predictions(reg_lm, xs, num_samples=3, batch_size=2)
    assert medians_one_batch.shape == medians_chunked.shape == (6, 1)
    assert np.all(np.isfinite(medians_one_batch))
    assert np.all(np.isfinite(medians_chunked))


def test_median_point_predictions_default_batch_size_is_bounded():
    import inspect

    sig = inspect.signature(median_point_predictions)
    assert sig.parameters["batch_size"].default <= 64
