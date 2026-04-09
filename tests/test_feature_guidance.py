from __future__ import annotations

from textwrap import dedent

from llamea.feature_guidance import compute_feature_guidance
from llamea.solution import Solution


def _make_solution(code: str, fitness: float) -> Solution:
    sol = Solution(code=dedent(code), name="Algo", description="Test algo")
    sol.set_scores(fitness, "feedback")
    return sol


def _make_feature_solution(features: dict[str, float], fitness: float) -> Solution:
    sol = Solution(code="pass", name="Algo", description="Feature test")
    sol.set_scores(fitness, "feedback")
    sol.add_metadata("ast_features", features)
    return sol


SIMPLE_BODY = """
            candidate = [0.0] * self.dim
            value = func(candidate)
            if value < best:
                best = value
"""

LOOPY_BODY = """
            candidate = [0.0] * self.dim
            for axis in range(self.dim):
                candidate[axis] = (-1) ** axis * (step + axis)
            value = func(candidate)
            if value < best:
                best = value
            if value < 0.0:
                best = value * 0.9
"""

NESTED_BODY = """
            candidate = [0.0] * self.dim
            for axis in range(self.dim):
                candidate[axis] = (-1) ** (axis + step) * (step + axis)
                if axis % 2 == 0:
                    candidate[axis] += 0.5
                else:
                    for inner in range(axis):
                        candidate[axis] -= inner * 0.1
            value = func(candidate)
            if value < best:
                best = value
            else:
                best = 0.5 * (best + value)
"""


def _wrap_body(body: str) -> str:
    return f"""
class Algo:
    def __init__(self, budget=5, dim=3):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best = float('inf')
        for step in range(self.budget):
{body}
        return best
"""


def test_feature_guidance_prefers_increasing_complexity():
    solutions = [
        _make_solution(_wrap_body(SIMPLE_BODY), 0.1),
        _make_solution(_wrap_body(LOOPY_BODY), 0.6),
        _make_solution(_wrap_body(NESTED_BODY), 1.2),
        _make_solution(_wrap_body(NESTED_BODY + SIMPLE_BODY), 1.5),
    ]
    print("Solutions: ", solutions)
    guidance = compute_feature_guidance(solutions, minimization=False)
    print('Guidance: ', guidance)
    assert guidance is not None
    assert guidance.action == "increase"
    assert guidance.message
    assert guidance.feature_name.lower() in guidance.message.lower()


def test_feature_guidance_accounts_for_minimization():
    solutions = [
        _make_solution(_wrap_body(SIMPLE_BODY), 0.05),
        _make_solution(_wrap_body(LOOPY_BODY), 0.5),
        _make_solution(_wrap_body(NESTED_BODY), 1.2),
        _make_solution(_wrap_body(LOOPY_BODY + SIMPLE_BODY), 0.8),
    ]

    guidance_max = compute_feature_guidance(solutions, minimization=False)
    guidance_min = compute_feature_guidance(solutions, minimization=True)

    assert guidance_max is not None
    assert guidance_min is not None
    assert guidance_max.action != guidance_min.action


def test_feature_guidance_message_used_in_prompt():
    class DummyLLM:
        def __init__(self) -> None:
            self.model = "dummy"

        def set_logger(self, logger):  # pragma: no cover - unused in tests
            self.logger = logger

        def sample_solution(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def query(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    def evaluator(solution: Solution, logger=None):
        solution.set_scores(0.0, "feedback")
        return solution

    from llamea.llamea import LLaMEA

    optimizer = LLaMEA(
        evaluator,
        llm=DummyLLM(),
        n_parents=1,
        n_offspring=1,
        log=False,
        feature_guided_mutation=True,
    )

    sol = _make_solution(_wrap_body(SIMPLE_BODY), 0.1)
    optimizer.population = [sol]
    optimizer.feature_guidance_message = "Increase testing metric"

    prompt = optimizer.construct_prompt(sol)
    assert isinstance(prompt, list)
    assert any("Increase testing metric" in item["content"] for item in prompt)


def test_parent_specific_shap_guidance_prefers_parent_action():
    solutions = [
        _make_feature_solution({"edges": 0.0}, 3.0),
        _make_feature_solution({"edges": 1.0}, 2.5),
        _make_feature_solution({"edges": 2.0}, 2.0),
        _make_feature_solution({"edges": 3.0}, 1.5),
    ]

    parent = _make_feature_solution({"edges": 5.0}, 1.0)

    guidance = compute_feature_guidance(
        solutions,
        minimization=False,
        parent=parent,
    )

    assert guidance is not None
    assert guidance.action == "decrease"
    assert guidance.feature_name == "edges"
    assert "parent" in guidance.message.lower()


def test_parent_guidance_handles_extreme_feature_values():
    solutions = [
        _make_feature_solution({"Nodes": 1.0, "Edges": 0.0}, 1.0),
        _make_feature_solution({"Nodes": 2.0, "Edges": 1.0}, 2.0),
        _make_feature_solution({"Nodes": 3.0, "Edges": 1.5}, 3.0),
        _make_feature_solution({"Nodes": 4.0, "Edges": 2.0}, 3.5),
    ]

    parent = _make_feature_solution({"Nodes": 10.0, "Edges": 5.0}, 0.5)

    guidance = compute_feature_guidance(
        solutions,
        minimization=False,
        parent=parent,
    )

    assert guidance is not None
    assert guidance.feature_name in {"Nodes", "Edges"}
    assert guidance.message
