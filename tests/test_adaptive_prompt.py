from unittest.mock import MagicMock

from llamea.llamea import LLaMEA
from llamea.solution import Solution


def f(individual, logger=None):
    individual.set_scores(1.0, "ok")
    return individual


class DummyLLM:
    def __init__(self):
        self.model = "dummy"
        self.query = MagicMock(return_value="Refined prompt")
        self.sample_solution = MagicMock(
            return_value=Solution(
                code="class Algo:\n    pass", name="Algo", description="desc"
            )
        )


def test_adaptive_prompt_updates_task_prompt():
    llm = DummyLLM()
    optimizer = LLaMEA(
        f,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        adaptive_prompt=True,
        log=False,
    )

    individual = Solution(
        code="class Base:\n    pass",
        name="Base",
        task_prompt="Original prompt",
    )
    individual.feedback = "Needs work"

    evolved = optimizer.evolve_solution(individual)

    assert evolved.task_prompt == "Refined prompt"
    llm.query.assert_called_once()
    session = llm.sample_solution.call_args[0][0]
    assert "Refined prompt" in session[0]["content"]
