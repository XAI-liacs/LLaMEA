# AGENTS

Welcome, Codex and other AI agents! This file contains instructions for working with the LLaMEA repository.

## Setup
- Use **Python 3.11+**.
- Install the [uv](https://github.com/astral-sh/uv) package manager if it is not already available.
- Install dependencies via `uv sync`.
  - Add `--group dev` for development tools and `--group docs` for documentation tools as needed.
- Run commands within the environment using `uv run <command>`.

## Development Tools
- Format code with:
  - `uv run isort llamea/`
  - `uv run black llamea/`
- Follow style guidelines:
  - 4-space indentation.
  - Limit lines to 80 characters.
  - Capitalize class names; use lowercase for function names.
- Build documentation when necessary with:
  - `uv run sphinx-apidoc -o docs/ llamea/`
  - `uv run sphinx-build -b html docs docs/_build`

## Testing
- Run the test suite before committing code changes:
  - `uv run pytest  tests/`
- Aim for at least 80% test coverage, using `uv run pytest --cov=llamea --cov-report=xml tests/` when measuring coverage.

## Pull Requests
- Ensure all tests pass and code is formatted before committing.
- Use clear, descriptive commit messages referencing related issues where applicable.

