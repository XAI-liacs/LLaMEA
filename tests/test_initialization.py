import pytest
from llamea import LLaMEA, Ollama_LLM


def f(ind, logger):
    return f"feedback {ind.name}", 1.0, ""


def test_default_initialization():
    """Test the default initialization of the LLaMEA class."""
    optimizer = LLaMEA(
        f,
        llm=Ollama_LLM("test_model"),
        task_prompt="test prompt",
        example_prompt="example prompt",
        output_format_prompt="format prompt",
        log=False,
    )
    assert optimizer.llm.model == "test_model"
    assert optimizer.task_prompt == "test prompt", "Task prompt should be set correctly"
    assert (
        optimizer.example_prompt == "example prompt"
    ), "Example prompt should be set correctly"
    assert (
        optimizer.output_format_prompt == "format prompt"
    ), "Output format prompt should be set correctly"


def test_custom_initialization():
    """Test custom initialization parameters."""
    optimizer = LLaMEA(f, llm=Ollama_LLM("custom model"), budget=500, log=False)
    assert optimizer.budget == 500, "Custom budget should be respected"
