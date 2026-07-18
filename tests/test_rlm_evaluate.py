import numpy as np
import pytest

from llamea.rlm_surrogate.data_pipeline import RLMExample
from llamea.rlm_surrogate.evaluate import (
    budget_reduction_simulation,
    evaluate_predictions,
    random_topk_overlap_expected,
    rank_metrics,
    topk_overlap,
    within_run_ranking,
)


def _example(id_, run_id, generation, fitness):
    return RLMExample(
        id=id_,
        run_id=run_id,
        generation=generation,
        parent_ids=[],
        x="code",
        y=fitness,
        fitness=fitness,
    )


def test_rank_metrics_perfect_correlation():
    y = [0.1, 0.5, 0.9, 0.3]
    m = rank_metrics(y, y)
    assert m["spearman_rho"] == pytest.approx(1.0)
    assert m["kendall_tau"] == pytest.approx(1.0)


def test_rank_metrics_inverse_correlation():
    y_true = [1, 2, 3, 4]
    y_pred = [4, 3, 2, 1]
    m = rank_metrics(y_true, y_pred)
    assert m["spearman_rho"] == pytest.approx(-1.0)


def test_rank_metrics_degenerate_constant_input():
    m = rank_metrics([1, 1, 1], [1, 2, 3])
    assert np.isnan(m["spearman_rho"])


def test_topk_overlap_perfect_and_worst_case():
    y_true = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    assert topk_overlap(y_true, y_true, k=2) == pytest.approx(1.0)
    y_pred_worst = -y_true  # exactly inverted ranking
    assert topk_overlap(y_true, y_pred_worst, k=2) == pytest.approx(0.0)


def test_random_topk_overlap_expected_matches_k_over_n():
    assert random_topk_overlap_expected(k=5, n=20) == pytest.approx(0.25)
    assert random_topk_overlap_expected(k=100, n=20) == 1.0  # clamped to n


def test_within_run_ranking_groups_by_run_and_respects_min_n():
    examples = [_example(str(i), "r1", 0, float(i)) for i in range(10)]
    examples += [_example(f"b{i}", "r2", 0, float(i)) for i in range(3)]  # below min_n
    y_pred = [e.fitness for e in examples]  # perfect predictions
    result = within_run_ranking(examples, y_pred, min_n=5)
    assert set(result["per_run"].keys()) == {"r1"}
    assert result["per_run"]["r1"]["spearman_rho"] == pytest.approx(1.0)
    assert result["per_run"]["r1"]["top1_hit"] is True
    assert result["n_runs_evaluated"] == 1


def test_within_run_ranking_empty_when_no_run_meets_min_n():
    examples = [_example(str(i), "r1", 0, float(i)) for i in range(3)]
    result = within_run_ranking(examples, [e.fitness for e in examples], min_n=5)
    assert result["per_run"] == {}
    assert result["n_runs_evaluated"] == 0


def test_budget_reduction_simulation_rlm_matches_all_when_k_over_n_is_1():
    examples = [_example(str(i), "r1", 0, float(i)) for i in range(10)]
    y_pred = [e.fitness for e in examples]
    result = budget_reduction_simulation(
        examples, y_pred, k_over_n_list=[1.0], min_batch_size=4, n_random_draws=20
    )
    entry = result["results"][0]
    assert entry["rlm_best_found_fraction"] == pytest.approx(1.0)
    assert entry["random_best_found_fraction"] == pytest.approx(1.0)
    assert entry["compute_savings"] == pytest.approx(0.0)


def test_budget_reduction_simulation_perfect_model_beats_random_at_low_k():
    rng = np.random.default_rng(0)
    fitness = rng.uniform(size=30)
    examples = [_example(str(i), "r1", 0, float(f)) for i, f in enumerate(fitness)]
    y_pred = fitness  # perfect ranker
    result = budget_reduction_simulation(
        examples,
        y_pred,
        k_over_n_list=[0.2],
        min_batch_size=4,
        n_random_draws=200,
        seed=0,
    )
    entry = result["results"][0]
    assert entry["rlm_best_found_fraction"] == pytest.approx(1.0)
    assert entry["rlm_best_found_fraction"] >= entry["random_best_found_fraction"]
    assert entry["compute_savings"] == pytest.approx(0.8)


def test_budget_reduction_simulation_skips_small_batches():
    examples = [
        _example(str(i), "r1", 0, float(i)) for i in range(3)
    ]  # batch too small
    result = budget_reduction_simulation(
        examples, [e.fitness for e in examples], min_batch_size=4
    )
    assert result["results"] == []
    assert result["n_batches_total"] == 0


def test_evaluate_predictions_end_to_end_pure():
    examples = [_example(str(i), "r1", i % 3, float(i)) for i in range(12)]
    y_pred = [e.fitness + 0.01 for e in examples]
    report = evaluate_predictions(examples, y_pred, label="test")
    assert report["overall"]["spearman_rho"] == pytest.approx(1.0)
    assert "by_run" in report
    assert "within_run_ranking" in report
    assert "budget_reduction" in report
