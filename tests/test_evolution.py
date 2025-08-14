from unittest.mock import MagicMock

import numpy as np
import pytest

from llamea import LLaMEA, Ollama_LLM


# Helper
class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)


f_try = 0


def f(ind, logger=None):
    global f_try
    f_try += 1
    ind.set_scores(f_try * 0.1, f"feedback {ind.name} {f_try}")
    return ind


def test_evolutionary_process():
    """Test the evolutionary process loop to ensure it updates generations."""
    global f_try
    f_try = 0
    response = "# Description: Long Example Algorithm\n# Code:\n```python\nclass ExampleAlgorithm:\n    pass\n```"
    optimizer = LLaMEA(
        f,
        n_parents=1,
        n_offspring=1,
        llm=Ollama_LLM("model"),
        experiment_name="test evolution",
        elitism=True,
        budget=10,
        log=True,
    )
    optimizer.llm.query = MagicMock(return_value=response)
    best_so_far = optimizer.run()  # Assuming run has a very simple loop for testing
    assert (
        best_so_far.code == "class ExampleAlgorithm:\n    pass"
    ), "best should be class ExampleAlgorithm(object):\n    pass"
    assert (
        best_so_far.fitness == 1.0
    ), f"Fitness should be 1.0, is {best_so_far.fitness}"
    assert (
        optimizer.generation == 10
    ), f"Generation should increment correctly, is {optimizer.generation}"


def test_evolutionary_process_with_errors():
    """Test the evolutionary process loop to ensure it updates generations even when errors occur."""
    global f_try
    f_try = 0
    response = "hi!"
    optimizer = LLaMEA(
        f,
        llm=Ollama_LLM("model"),
        n_parents=1,
        n_offspring=1,
        role_prompt="You are super cute",
        task_prompt="just say hi",
        experiment_name="test evolution with errors",
        budget=10,
        log=False,
    )
    optimizer.llm.query = MagicMock(return_value=response)
    best_so_far = optimizer.run()  # Assuming run has a very simple loop for testing
    assert (
        best_so_far.fitness == -np.inf
    ), f"Fitness should be 0.0 is {best_so_far.fitness}"
    assert optimizer.generation == 10, "Generation should increment correctly"


def test_population_evaluation_mode():
    """Ensure evaluation function can process a population at once."""

    def f_pop(population, parents=None, logger=None):
        for i, ind in enumerate(population):
            ind.set_scores(float(i), f"feedback {i}")
        return population, parents

    response = "# Description: Algo\n# Code:\n```python\nclass Algo:\n    pass\n```"

    optimizer = LLaMEA(
        f_pop,
        n_parents=2,
        n_offspring=2,
        llm=Ollama_LLM("model"),
        experiment_name="population mode",
        budget=4,
        evaluate_population=True,
        log=False,
    )
    optimizer.llm.query = MagicMock(return_value=response)
    best = optimizer.run()

    assert len(optimizer.run_history) == 4
    assert best.fitness in [0.0, 1.0]
