# LLaMEA – Claude Code Guide

## Repository overview

LLaMEA is a Python framework that drives an evolutionary loop powered by an LLM.
It generates, evaluates, and refines code (e.g. optimisation algorithms, ML pipelines)
by prompting the LLM to mutate the current best solution.

Key source files under `llamea/`:
- `llamea.py` – main `LLaMEA` class; evolutionary loop, prompt construction, default prompts
- `llm.py` – LLM wrappers (`OpenAI_LLM`, `Gemini_LLM`, `Ollama_LLM`, `Dummy_LLM`, …)
- `solution.py` – `Solution` dataclass holding code, description, score, feedback
- `diffmodemanager.py` – applies SEARCH/REPLACE diffs returned by the LLM
- `feature_guidance.py` – complexity-based mutation guidance
- `loggers.py` / `utils.py` – experiment logging and helper utilities

Examples live in `examples/`, benchmarks in `benchmarks/`.

## Development workflow

### Install dependencies (always do this first)
```bash
uv sync --dev
```

### Run import sorter (run before black)
```bash
uv run isort llamea/
```

### Run formatter
```bash
uv run black llamea/
```

### Run tests
```bash
uv run pytest
# with coverage:
uv run pytest --cov=llamea
```

### Typical pre-commit sequence
```bash
uv sync --dev
uv run isort llamea/
uv run black llamea/
uv run pytest
```

## Code conventions

- **Formatter**: `black` (line length default, ~88 chars)
- **Import order**: `isort` (stdlib → third-party → local)
- **Python**: ≥ 3.11
- **Multi-line prompt strings**: wrap with `textwrap.dedent()` to avoid sending
  accidental leading whitespace to the LLM, e.g.:
  ```python
  import textwrap
  prompt = textwrap.dedent("""
      Your instructions here.
      More lines.
      """)
  ```

## Prompt architecture

`LLaMEA.__init__` accepts four prompt parameters (all have defaults):
| Parameter | Purpose |
|---|---|
| `role_prompt` | LLM system persona (neutral, task-independent) |
| `task_prompt` | Problem description; what the LLM should produce |
| `example_prompt` | Illustrative code example |
| `output_format_prompt` | Required response format |

A fifth prompt, `diff_output_format_prompt`, is built-in and controls the
SEARCH/REPLACE diff mode.

The default `role_prompt` is intentionally generic ("You are a highly skilled
computer scientist and Python expert.") so it works across domains without
users needing to override it.

## Git branch convention

Feature branches follow the pattern `feature/<short-description>`.
