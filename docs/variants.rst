LLaMEA variants
===============

The configurations studied in our papers are available as ready-made
classes in :mod:`llamea.variants`. Each is a thin subclass of
:class:`~llamea.llamea.LLaMEA` that only changes default arguments, so every
LLaMEA argument still works and **overrides the preset**.

.. list-table::
   :header-rows: 1
   :widths: 16 40 26 18

   * - Class
     - What it does
     - Preset
     - Paper
   * - :class:`~llamea.llamea.LLaMEA`
     - The base algorithm.
     - ``(5+5)``, one mutation operator
     - `IEEE TEVC 2025 <https://doi.org/10.1109/TEVC.2024.3497793>`_
   * - :class:`~llamea.variants.LLaMEA_HPO`
     - The LLM writes the algorithm and a ConfigSpace of its
       hyper-parameters; your evaluation tunes them (e.g. with SMAC).
     - ``HPO=True``, ``(5+5)``, "refine or redesign" prompt
     - `ACM TELO 2025 <https://doi.org/10.1145/3731567>`_
   * - :class:`~llamea.variants.LLaMEA_BO`
     - Evolves complete Bayesian optimisation algorithms.
     - ``(4+16)``, crossover 0.6 / mutation 0.4, budget 100, BO prompt
     - `arXiv:2505.21034 <https://arxiv.org/abs/2505.21034>`_
   * - :class:`~llamea.variants.LLaMEA_MO`
     - Several objectives; NSGA-II-style selection; returns a Pareto front.
     - ``multi_objective=True``, ``(5+10)``
     - –
   * - :class:`~llamea.variants.LLaMEA_SAGE`
     - Mutations guided by an explainable-AI analysis (XGBoost + TreeSHAP)
       of code features.
     - ``feature_guided_mutation=True``
     - `arXiv:2601.21511 <https://arxiv.org/abs/2601.21511>`_

Usage
-----

.. code-block:: python

   from llamea import LLaMEA_BO, LLaMEA_HPO, LLaMEA_MO, get_variant

   best = LLaMEA_BO(evaluate, llm).run()                 # paper defaults
   best = LLaMEA_BO(evaluate, llm, budget=200).run()     # override any argument

   front = LLaMEA_MO(evaluate_mo, llm,
                     objectives=["Distance", "Fuel"],
                     task_prompt="...").run()            # list of solutions

   Variant = get_variant("hpo")                          # by name, e.g. from a config
   optimizer = Variant(evaluate, llm, budget=50)

``get_variant`` accepts the short name (``"llamea"``, ``"hpo"``, ``"bo"``,
``"mo"``, ``"sage"``) or the class name (``"LLaMEA-BO"``, ``"LLaMEA_BO"``).

LLaMEA-HPO
----------

With ``HPO=True`` the output format asks for a ``Space: {...}`` dictionary
next to the code; LLaMEA turns it into a ``ConfigSpace.ConfigurationSpace``
on ``solution.configspace``. Your evaluation function tunes the algorithm on
training instances (for example with SMAC), then scores the best
configuration and reports that score. Offloading numeric tuning lets the LLM
spend its queries on structural changes. A complete set-up is in
``examples/black-box-opt-with-HPO.py``.

LLaMEA-BO
---------

The preset follows the paper: an elitist (4+16) strategy, crossover of two
parents with probability 0.6 (mutation otherwise), and a budget of 100
candidates. The task prompt asks for the core BO components (initial design,
surrogate model, acquisition function) with the BBOB interface
``__init__(budget, dim)`` / ``__call__(func)`` on ``[-5, 5]^d``, and the
example prompt is a small Gaussian-process BO with expected improvement. The
paper scores candidates by AOCC on BBOB functions in dimension 5 with
20 x dim evaluations. Pass ``task_prompt`` and ``example_prompt`` for another
interface.

LLaMEA-MO
---------

The evaluation function sets one value per objective:

.. code-block:: python

   from llamea import Fitness

   def evaluate_mo(solution, logger=None):
       distance, fuel = run(solution)
       solution.set_scores(Fitness({"Distance": distance, "Fuel": fuel}),
                           feedback=f"distance {distance:.1f}, fuel {fuel:.1f}")
       return solution

Survivors are chosen by non-dominated sorting and crowding distance, and
:meth:`~llamea.llamea.LLaMEA.run` returns the Pareto front. ``minimization``
applies to all objectives. See ``examples/multi_objective.py``.

LLaMEA-SAGE
-----------

Requires the optional extra ``pip install "llamea[feature-guidance]"``. After
every generation a gradient-boosted model is fitted on AST and complexity
features of the archive; TreeSHAP identifies the most influential feature,
and the mutation prompt asks the LLM to increase or decrease it.
Single-objective only.

Running variants in BLADE
-------------------------

To benchmark a variant against other methods, pass the same arguments to
BLADE's ``llamea`` method (see :doc:`blade`):

.. code-block:: yaml

   methods:
     - type: llamea
       name: LLaMEA-HPO
       llm: gemini
       params: {HPO: true, n_parents: 5, n_offspring: 5}

API
---

See :doc:`api_variants`.
