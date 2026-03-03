from unittest.mock import MagicMock

import pytest

from llamea import LLaMEA, Ollama_LLM, Solution


# Helper
class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)


def test_algorithm_generation():
    """Test the algorithm generation process."""

    def f(solution):
        return f"feedback {solution.name}", 1.0, "", {}

    optimizer = LLaMEA(
        f, llm=Ollama_LLM("model"), experiment_name="test generation", log=False
    )
    response = "# Description: Long Example Algorithm\n# Code:\n```python\nclass ExampleAlgorithm:\n    pass\n```"
    optimizer.llm.query = MagicMock(return_value=response)

    individual = optimizer.llm.sample_solution(
        session_messages=[{"role": "system", "content": "test prompt"}]
    )

    assert (
        individual.description == "Long Example Algorithm"
    ), f"Algorithm long name should be extracted correctly, is {individual.description}"
    assert (
        individual.name == "ExampleAlgorithm"
    ), "Algorithm name should be extracted correctly"
    assert (
        "class ExampleAlgorithm" in individual.code
    ), "Algorithm code should be extracted correctly"


def test_evolve_solution_with_diff():
    def f(sol, logger):
        return sol

    optimizer = LLaMEA(
        f,
        llm=Ollama_LLM("model"),
        experiment_name="diff",
        log=False,
        diff_mode=True,
        evaluate_population=False,
    )

    base = Solution(code="class MyAlgo:\n    pass\n", name="MyAlgo", description="d")
    optimizer.population = [base]
    diff_reply = """
```
<<<<<<< SEARCH
    pass
=======
    # Added loops
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE
```"""
    optimizer.llm.query = MagicMock(return_value=diff_reply)
    evolved = optimizer.evolve_solution(base)
    assert "# Added loops" in evolved.code, evolved.code + " does not contain added code '# Added loops'"
    assert "pass" not in evolved.code, evolved.code + " contains code 'pass'"
