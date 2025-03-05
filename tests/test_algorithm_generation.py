import pytest
from unittest.mock import MagicMock
from llamea import LLaMEA, Ollama_LLM

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
    optimizer.client.query = MagicMock(return_value=response)

    individual = optimizer.llm(
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
