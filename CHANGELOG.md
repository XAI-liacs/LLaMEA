# Changelog

## Unreleased

### Changed
- **Dependency ranges widened to the current major versions**:
  `openai` <4 (locks 3.59.0), `google-genai` <4 (3.6.1), `xgboost` <4 (3.4.1;
  3.2.0 on Python 3.11). Dev: `isort` <10, `pytest-cov` <8, `ipykernel` <8,
  `ioh` <0.5. Docs: `sphinx` >=7.4.7,<10 (9.2.0; 9.0.4 on Python 3.11),
  `myst-parser` <6, `furo` and `sphinx-autobuild` uncapped,
  `sphinxext-opengraph` <0.14. Examples: `pandas` <4. LLM extras:
  `transformers` <6. Lower bounds are unchanged, so existing installs keep
  resolving.
- Lockfile refreshed with `uv lock --upgrade` (e.g. `anthropic` 2.14.0,
  `torch` 2.15.0, `transformers` 5.24.1, `pandas` 3.1.1). `fonttools` is held
  at 4.68.0 because 4.69.0 ships only cp311 wheels.
- `Gemini_LLM` now pins the Gemini `v1beta` API. google-genai 3 switched its
  default to `v1`, which lacks some models (e.g. the `*-latest` aliases);
  set `GOOGLE_GENAI_API_VERSION` to choose another version.
- `OpenAI_LLM` also silences the `httpx2` logger, the HTTP client of openai 3.
- **`iohinspector` 0.0.3 → 0.3.0** (`examples` group, now `>=0.3.0,<0.4`).
  0.0.3 no longer worked with current polars (`plot_ecdf` crashed). The MA-BBOB
  `GettingStarted.ipynb` now uses `iohinspector.plots.plot_ecdf(f_min=,
  f_max=)` (was `plot.plot_ecdf(y_min=, y_max=)`) and `get_aocc(by=...)`
  (was `group_cols=`). AOCC values shift by at most 2e-3 because 0.3.0
  integrates from evaluation 1, which matches LLaMEA's own `correct_aoc` more
  closely; rankings are unchanged. ECDF plots now honour `f_max=100` (0.0.3
  silently used 1e8).

- **`pymoosh` 3.2 → 4.0.1** (`examples` group, now `>=4.0.1,<5`). 3.2's
  `photo()` calls `np.trapz`, which numpy 2.4 removed, so the photovoltaic
  benchmark crashed on the locked numpy. The photonic benchmarks now import
  `coefficient_I` from `PyMoosh.alt_methods` (it returns `r, t, R, T`) and
  `photo` from `PyMoosh.photo`; `sophisticated_antireflection_design` aliases
  `np.trapz` to `np.trapezoid` because PyMoosh 4.0.1 still calls it. On 200
  fixed designs per problem, Bragg, mini-Bragg and photovoltaic costs match
  3.2 to 2e-15 (relative) and ellipsometry to 4e-8 (PyMoosh 4 refined its
  Gold model); rankings are identical.
- **`modcma`** is now `>=1.2.0,<2` and only installed on Python < 3.13. 1.2.0
  is the latest release; it has no cp313 wheel and its sdist cannot build,
  which made the `examples` group uninstallable on Python 3.13.

### Fixes
- **`Gemini_LLM.query` failed on multi-turn conversations.** Earlier messages
  were sent as plain-string parts, which google-genai rejects (1.x and 3.x),
  and the `assistant` role was not mapped to Gemini's `model`. History is now
  built as `{"role": "user"|"model", "parts": [{"text": ...}]}`, and `system`
  messages become the `system_instruction`. LLaMEA's own loop sends
  single-message sessions and was unaffected.
- The `apply_code_delta` docstring is now valid reStructuredText, so it no
  longer produces Sphinx errors.

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
- `certifi` 2026.10.19, `jupyterlab` 4.6.4, `pillow` 13.0.0 (#153).

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
- **Multi-objective selection was wrong when maximising** (#157, fixes #156).
  The minimisation fronts from `NonDominatedSorting` were reversed, which does
  not give the maximisation fronts, so a dominated solution could survive over
  a dominating one. Objectives are now negated instead, as `ParetoArchive`
  does. The new population also has `n_parents` members again, not
  `n_offspring`.
- **Multi-objective operator rewards** (#159, fixes #158). An offspring that
  neither dominates nor is dominated by its parent scored -1 when minimising
  and +1 when maximising; it now scores 0. Scalar fitness is unchanged.

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
