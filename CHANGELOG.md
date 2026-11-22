# Changelog

## 1.4.3 (2026-11-22)

### Changed
- **Minimum Python is now 3.11.5** (was 3.11). `soupsieve` releases that fix
  its open advisories require 3.11.5, so the lockfile previously kept a
  vulnerable 2.x for Python 3.11.0–3.11.4.

### Security
- **Dependency upgrades for known vulnerabilities** in the lockfile:
  `black` 26.10.1, `filelock` 4.3.0, `jupyter-server` 2.21.1, `msgpack` 1.2.4,
  `nbconvert` 7.17.2, `opentelemetry-api` 1.45.2, `pyasn1` 0.6.5,
  `setuptools` 84.2.1, `soupsieve` 4.1.3, `tornado` 6.6.1.
- The `examples` group pinned `scikit-learn==1.3.0` (GHSA-jw8x-6495-233v); it
  now requires `scikit-learn>=1.5.0,<2` and locks 1.9.1.
- `polars` 1.44.0 was yanked upstream; locked 1.44.2 instead.

### Fixes
- **Crossover operators crashed with `parent_selection="random"`** (the
  default). `LLaMEA._select_parents` returned a NumPy object array for random
  selection, so `_select_parents(...) + [individual]` in `_evolve_solution`
  made NumPy add `Solution` objects element-wise (`TypeError: unsupported
  operand type(s) for +: 'Solution' and 'Solution'`). It now returns a plain
  `list`, like roulette and tournament selection, with the same sampling with
  replacement. Downstream workarounds that require `parent_selection:
  tournament` for crossover are no longer needed.
- **Crossover offspring recorded a non-existent parent id.** The selected
  individual's id was taken from its internal copy, which gets a fresh id and
  is never logged, so lineage in `parent_ids` (and `operator`) pointed nowhere.
  Crossover offspring now record the original individual's id, as mutation
  offspring already did.

## 1.4.2 (2026-10-23)

### Changed
- **`anthropic` upgraded to 2.x** (#154). LLaMEA now requires
  `anthropic>=2.1.0,<3`, which uses Claude API version `2026-10-21` and is built
  on `httpx2`.
- **`Anthropic_LLM` no longer sends `temperature`** (#154). API version
  `2026-10-21` rejects sampling parameters on every model. The argument is kept
  (default `None`) so existing code still runs; passing a value raises a
  `DeprecationWarning` and is ignored.

### Added
- `Anthropic_LLM(effort=...)` sets `output_config.effort` (`"low"` to `"max"`)
  (#154).
- `Anthropic_LLM(fallbacks=...)` enables server-side refusal fallbacks, e.g.
  `"default"`, so a refused request is retried on a fallback model (#154).

### Fixes
- A refused Claude response (`stop_reason == "refusal"`) now returns an empty
  string instead of partial text (#154).
- `Anthropic_LLM` instances pickled with 1.4.1 or earlier load correctly (#154).

## 1.4.1 (2026-10-22)

Bug-fix release.

### Fixes
- **Operator prompts were swapped with their names** (#151). `LLaMEA` built the
  default operator, and operators given as plain strings, with positional
  arguments in the wrong order, so the LLM received the literal word
  `"mutation"` instead of the mutation prompt. Operators are now built with
  keyword arguments.
- **A single operator is accepted** (#151). Passing one `str` or `Operator` as
  `operators` no longer splits a string into one operator per character.
- **`Solution.operator_id` is kept** (#151). It is now set in `__init__`,
  included in `to_dict()` and `copy()`, and survives pickling (older pickles
  default it).
- **Crossover example and docs** (#151). `examples/crossover.py` and
  `docs/crossover.rst` used the wrong `Operator` argument order, class name and
  had a syntax error.

### Added
- `Anthropic_LLM` wrapper for Claude models (#149).

### Dependencies
- `werkzeug` 3.1.8 → 3.1.9 (#150).
