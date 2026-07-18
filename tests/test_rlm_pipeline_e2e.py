"""End-to-end smoke test of the full RLM surrogate pipeline (data -> train ->
evaluate -> report) against the synthetic fixtures, using the fully-offline
tiny `char`-encoder config so it needs no network access and no gated
HuggingFace model. Skipped automatically if torch/regress_lm aren't
installed (the `rlm-surrogate` dependency group)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("regress_lm")

from llamea.rlm_surrogate.config import RLMSurrogateConfig
from llamea.rlm_surrogate.data_pipeline import run_pipeline
from llamea.rlm_surrogate.evaluate import run_full_evaluation
from llamea.rlm_surrogate.report import build_report
from llamea.rlm_surrogate.train import train

FIXTURES = Path(__file__).parent / "fixtures" / "rlm"
TINY_CONFIG = (
    Path(__file__).parent.parent
    / "llamea"
    / "rlm_surrogate"
    / "configs"
    / "tiny_local_test.yaml"
)


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "ckpt"
    eval_dir = tmp_path / "eval"
    report_dir = tmp_path / "report"

    stats = run_pipeline(FIXTURES, data_dir)
    assert stats["n_examples_total"] > 0

    config = RLMSurrogateConfig.from_yaml(TINY_CONFIG)
    manifest = train(config, data_dir / "train.jsonl", data_dir / "val.jsonl", ckpt_dir)
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "config.yaml").exists()
    assert manifest["n_train"] == stats["split"]["n_train"]

    eval_results = run_full_evaluation(
        ckpt_dir, data_dir / "train.jsonl", data_dir / "test.jsonl"
    )
    assert eval_results["arms"]["rlm"]["overall"]["n"] == stats["split"]["n_test"]
    assert "feature_baseline" in eval_results["arms"]
    assert "random" in eval_results["arms"]

    import json

    import yaml

    report_dir.mkdir(parents=True, exist_ok=True)
    model_config = yaml.safe_load((ckpt_dir / "config.yaml").read_text())
    report_text = build_report(stats, eval_results, model_config, {}, report_dir)
    assert (report_dir / "report.md").exists()
    assert "Recommendation" in report_text
    assert "Budget-reduction" in report_text
