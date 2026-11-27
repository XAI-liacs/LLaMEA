"""Ready-made configurations ("variants") of LLaMEA from our papers.

Each variant is a thin subclass of :class:`~llamea.llamea.LLaMEA` that only
changes default arguments, so every LLaMEA argument can still be passed and
overrides the preset::

    from llamea import LLaMEA_BO, Gemini_LLM

    llm = Gemini_LLM(api_key, "gemini-2.0-flash")
    optimizer = LLaMEA_BO(evaluate, llm, budget=100)
    best = optimizer.run()

Use :func:`get_variant` to look a variant up by its short name (``"hpo"``,
``"bo"``, ``"mo"``, ``"sage"``), e.g. from a configuration file.

To benchmark a variant against other LLM-driven design methods, run it in
BLADE (https://github.com/XAI-liacs/BLADE), passing the same arguments to
BLADE's ``LLaMEA`` method.
"""

from __future__ import annotations

import textwrap
import warnings
from typing import Any

from .llamea import LLaMEA
from .operator import Operator

__all__ = [
    "LLaMEA_HPO",
    "LLaMEA_BO",
    "LLaMEA_MO",
    "LLaMEA_SAGE",
    "VARIANTS",
    "get_variant",
]


class _Variant(LLaMEA):
    """Base class: merges :meth:`default_config` under the user's arguments."""

    #: Short registry name, e.g. ``"bo"``.
    key: str = ""
    #: Paper that introduced the variant.
    paper_title: str = ""
    paper_url: str = ""

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """The preset arguments (fresh objects on every call)."""
        return {}

    def __init__(self, f, llm, **kwargs):
        config = self.default_config()
        config.update(kwargs)
        super().__init__(f, llm, **config)


# --------------------------------------------------------------------- HPO

HPO_MUTATION_PROMPT = (
    "Either refine or redesign to improve the solution "
    "(and give it a distinct one-line description)."
)


class LLaMEA_HPO(_Variant):
    """LLaMEA-HPO: in-the-loop hyper-parameter optimisation.

    The LLM writes the algorithm *and* a ConfigSpace dictionary of its
    hyper-parameters; your evaluation function tunes them (e.g. with SMAC)
    before scoring the algorithm. The LLM can then spend its queries on
    structural changes instead of numeric tuning.

    Preset: ``HPO=True``, an elitist (5+5) strategy and the
    "refine or redesign" mutation prompt.

    Your evaluation function receives ``solution.configspace``
    (a ``ConfigSpace.ConfigurationSpace`` or ``None``). See
    ``examples/black-box-opt-with-HPO.py`` for a complete SMAC set-up.

    Paper: van Stein, Vermetten & Bäck, *In-the-loop Hyper-Parameter
    Optimization for LLM-Based Automated Design of Heuristics*, ACM TELO
    (2025), https://doi.org/10.1145/3731567.
    """

    key = "hpo"
    paper_title = (
        "In-the-loop Hyper-Parameter Optimization for LLM-Based Automated "
        "Design of Heuristics"
    )
    paper_url = "https://doi.org/10.1145/3731567"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "HPO": True,
            "n_parents": 5,
            "n_offspring": 5,
            "elitism": True,
            "operators": [Operator(HPO_MUTATION_PROMPT, name="mutation", weight=1.0)],
        }


# ---------------------------------------------------------------------- BO

BO_TASK_PROMPT = textwrap.dedent("""
    Your task is to design a novel and efficient Bayesian optimization (BO)
    algorithm in Python for expensive black-box optimization. The algorithm is
    evaluated on noiseless BBOB functions with a small evaluation budget
    (e.g. 20 x dim evaluations). The search space is [-5.0, 5.0]^dim
    (available as `func.bounds.lb` and `func.bounds.ub`).

    The code must contain a class with `__init__(self, budget, dim)` and
    `__call__(self, func)`, which minimizes `func` using at most `self.budget`
    function evaluations and returns the best function value and solution.
    Do not add other arguments without a default value.

    The algorithm should contain the core BO components: an initial design,
    a surrogate model and an acquisition function (with its optimization).
    You may use numpy, scipy and scikit-learn.
    """)

BO_EXAMPLE_PROMPT = textwrap.dedent("""
    An example of such code (a simple Gaussian-process BO with expected
    improvement), is as follows:
    ```python
    import numpy as np
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern


    class SimpleBO:
        def __init__(self, budget=100, dim=5):
            self.budget = budget
            self.dim = dim
            self.n_init = min(2 * dim, budget)

        def __call__(self, func):
            lb, ub = func.bounds.lb, func.bounds.ub
            X = np.random.uniform(lb, ub, size=(self.n_init, self.dim))
            y = np.array([func(x) for x in X])
            gp = GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True)
            while len(y) < self.budget:
                gp.fit(X, y)
                cand = np.random.uniform(lb, ub, size=(1000, self.dim))
                mu, sigma = gp.predict(cand, return_std=True)
                imp = y.min() - mu
                z = imp / np.maximum(sigma, 1e-12)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
                x = cand[np.argmax(ei)]
                X = np.vstack([X, x])
                y = np.append(y, func(x))
            best = np.argmin(y)
            return y[best], X[best]
    ```
    """)

BO_MUTATION_PROMPT = "Refine the strategy of the selected solution to improve it."
BO_CROSSOVER_PROMPT = (
    "Combine the selected solutions into one new Bayesian optimization "
    "algorithm that keeps their best ideas, and refine the result."
)


class LLaMEA_BO(_Variant):
    """LLaMEA-BO: evolve complete Bayesian optimisation algorithms.

    Preset (following the paper): an elitist (4+16) strategy, crossover of
    two parents with probability 0.6 and mutation otherwise, a budget of
    100 candidates, and a task prompt and example that ask for the core BO
    components (initial design, surrogate model, acquisition function).

    The default prompts target BBOB-style problems (``func.bounds``,
    ``__init__(budget, dim)``, ``__call__(func)``); pass ``task_prompt`` and
    ``example_prompt`` for another interface. Your evaluation function scores
    each algorithm (the paper uses AOCC on BBOB functions in dimension 5 with
    a budget of 20 x dim evaluations).

    Paper: Li, van Stein, Bäck & Raponi, *LLaMEA-BO: A Large Language Model
    Evolutionary Algorithm for Automatically Generating Bayesian Optimization
    Algorithms*, arXiv:2505.21034 (2025).
    """

    key = "bo"
    paper_title = (
        "LLaMEA-BO: A Large Language Model Evolutionary Algorithm for "
        "Automatically Generating Bayesian Optimization Algorithms"
    )
    paper_url = "https://arxiv.org/abs/2505.21034"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "n_parents": 4,
            "n_offspring": 16,
            "elitism": True,
            "budget": 100,
            "task_prompt": BO_TASK_PROMPT,
            "example_prompt": BO_EXAMPLE_PROMPT,
            "minimization": False,
            "operators": [
                Operator(BO_MUTATION_PROMPT, name="mutation", weight=0.4),
                Operator(
                    BO_CROSSOVER_PROMPT,
                    name="crossover",
                    weight=0.6,
                    number_of_parents=2,
                ),
            ],
        }


# ---------------------------------------------------------------------- MO


class LLaMEA_MO(_Variant):
    """LLaMEA-MO: algorithm design with several objectives.

    Survivor selection uses non-dominated sorting and crowding distance
    (NSGA-II style) on a :class:`~llamea.multi_objective_fitness.Fitness`
    per solution, and :meth:`run` returns the Pareto front (a list of
    solutions) instead of a single best solution.

    Args:
        f: Evaluation function. It must call
            ``solution.set_scores(Fitness({...}), feedback)`` with one value
            per objective.
        llm: The LLM to use.
        objectives: Objective names, e.g. ``["Distance", "Fuel"]`` (alias of
            ``multi_objective_keys``).
        **kwargs: Any other :class:`~llamea.llamea.LLaMEA` argument.
            ``minimization`` applies to all objectives.

    Preset: ``multi_objective=True`` and an elitist (5+10) strategy, which
    gives the non-dominated sorting a front to work with. There is no
    sensible default task prompt for an unknown multi-objective problem, so
    pass ``task_prompt`` (and ``example_prompt``).
    """

    key = "mo"
    paper_title = ""
    paper_url = ""

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "multi_objective": True,
            "n_parents": 5,
            "n_offspring": 10,
            "elitism": True,
        }

    def __init__(self, f, llm, objectives: list[str] | None = None, **kwargs):
        if objectives is not None:
            kwargs["multi_objective_keys"] = list(objectives)
        if not kwargs.get("multi_objective_keys"):
            raise ValueError(
                "LLaMEA_MO needs the objective names, e.g. "
                "objectives=['Distance', 'Fuel']."
            )
        if not kwargs.get("task_prompt"):
            warnings.warn(
                "LLaMEA_MO without a task_prompt uses the single-objective "
                "BBOB task prompt; describe your problem and objectives.",
                stacklevel=2,
            )
        super().__init__(f, llm, **kwargs)


# -------------------------------------------------------------------- SAGE


class LLaMEA_SAGE(_Variant):
    """LLaMEA-SAGE: mutation guided by structural feedback from explainable AI.

    A surrogate (XGBoost) learns how AST and code-complexity features of the
    archive relate to fitness; TreeSHAP picks the feature that matters most,
    and the mutation prompt asks the LLM to increase or decrease it.

    Preset: ``feature_guided_mutation=True``. Requires the optional extra:
    ``pip install "llamea[feature-guidance]"``. Single-objective only.

    Paper: *LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural
    Feedback from Explainable AI*, arXiv:2601.21511 (2026).
    """

    key = "sage"
    paper_title = (
        "LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural "
        "Feedback from Explainable AI"
    )
    paper_url = "https://arxiv.org/abs/2601.21511"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"feature_guided_mutation": True}


# ---------------------------------------------------------------- registry

#: Short name -> class. ``"llamea"`` is the plain algorithm.
VARIANTS: dict[str, type[LLaMEA]] = {
    "llamea": LLaMEA,
    "hpo": LLaMEA_HPO,
    "bo": LLaMEA_BO,
    "mo": LLaMEA_MO,
    "sage": LLaMEA_SAGE,
}


def get_variant(name: str) -> type[LLaMEA]:
    """Return the LLaMEA class for a variant name.

    Accepts the short name (``"bo"``) or the class name (``"LLaMEA-BO"``,
    ``"LLaMEA_BO"``), case-insensitively.

    Raises:
        KeyError: For an unknown name; the message lists the known ones.
    """
    key = name.strip().lower().replace("-", "_")
    if key.startswith("llamea_"):
        key = key[len("llamea_") :]
    if key not in VARIANTS:
        raise KeyError(
            f"Unknown LLaMEA variant {name!r}; choose one of {sorted(VARIANTS)}."
        )
    return VARIANTS[key]
