import pytest
from llamea import LLaMEA, Ollama_LLM


def f(ind, logger):
    return f"feedback {ind.name}", 1.0, ""


def test_default_initialization():
    """Test the default initialization of the LLaMEA class."""
    optimizer = LLaMEA(f, llm=Ollama_LLM("test_model"))
    assert optimizer.client.model == "test_model"


def test_custom_initialization():
    """Test custom initialization parameters."""
    optimizer = LLaMEA(f, llm=Ollama_LLM("custom model"), budget=500, log=False)
    assert optimizer.budget == 500, "Custom budget should be respected"
