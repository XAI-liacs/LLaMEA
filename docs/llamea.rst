Configuring LLaMEA
==================

The :class:`~llamea.llamea.LLaMEA` class implements the evolutionary loop
around a large language model. This page groups its arguments by purpose;
the :doc:`API reference <api_llamea>` lists them all. For ready-made
combinations from our papers, see :doc:`variants`.

Features
--------

* **Niching**: ``niching="sharing"``, ``"clearing"``, ``"novelty"`` or
  ``"map_elites"`` keeps the population diverse. ``distance_metric``,
  ``niche_radius``, ``adaptive_niche_radius``, ``clearing_interval``,
  ``behavior_descriptor``, ``map_elites_bins``, ``novelty_k`` and
  ``novelty_archive_size`` tune them. Niching changes ``solution.fitness``
  for selection only; the raw fitness is kept in
  ``solution.metadata["raw_fitness"]`` and the best solution is judged on it.
* **Diff mode**: ``diff_mode=True`` asks for SEARCH/REPLACE patches instead
  of complete files, which saves tokens on large code. A reply whose SEARCH
  blocks match nothing is reported back to the LLM as an error.
* **Population evaluation**: with ``evaluate_population=True``, ``f``
  receives the offspring and the parents,
  ``f(offspring, parents, logger) -> (offspring, parents)``, for batch or
  relative evaluations.
* **Multi-objective mode**: ``multi_objective=True`` with
  ``multi_objective_keys=[...]``; see :class:`~llamea.variants.LLaMEA_MO`.
* **Adaptive mutation**: ``adaptive_mutation=True`` replaces the mutation
  prompt with one that fixes the fraction of lines the LLM may change, drawn
  from a heavy-tailed distribution (Yin et al., 2025).
* **Adaptive prompt**: ``adaptive_prompt=True`` lets the LLM rewrite the task
  prompt before each mutation, so the prompt co-evolves with the solutions.
* **Feature-guided mutation**: ``feature_guided_mutation=True``, see
  :class:`~llamea.variants.LLaMEA_SAGE`.
* **Timeouts**: an evaluation that does not return within ``eval_timeout``
  seconds (plus a margin) is recorded as a failed candidate with a
  ``TimeoutError``; it counts towards the budget.
* **Warm start**: after every generation LLaMEA stores itself in
  ``<log directory>/llamea_config.pkl``.
  ``LLaMEA.warm_start("<log directory>")`` restores the object; call
  ``.run()`` on it to continue in the same directory.
* **Initial population**: ``optimizer.run("<log directory>")`` starts a new
  run from the last ``n_parents`` solutions logged in an earlier run. Use the
  same task set-up as that run.

Arguments
---------

The most important keyword arguments of :class:`LLaMEA` are summarised below.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning
   * - ``f``
     - Evaluation function returning feedback, fitness and error.
   * - ``llm``
     - Language model wrapper used for generation.
   * - ``n_parents`` / ``n_offspring``
     - Number of parents and offspring per generation.
   * - ``elitism``
     - ``True`` uses a ``(μ+λ)`` strategy, ``False`` a ``(μ,λ)`` strategy.
   * - ``role_prompt`` / ``task_prompt`` / ``example_prompt`` / ``output_format_prompt``
     - Prompt engineering controls.
   * - ``operators`` / ``operator_weight_updater`` / ``parent_selection`` / ``tournament_size``
     - Mutation and crossover prompts and how they and the parents are chosen (:doc:`crossover`).
   * - ``adaptive_mutation`` / ``adaptive_prompt`` / ``feature_guided_mutation``
     - Mutation-rate control, co-evolving task prompt, explainable-AI guidance.
   * - ``multi_objective`` / ``multi_objective_keys``
     - Enable Pareto-based optimization and define objective names.
   * - ``budget`` / ``eval_timeout`` / ``max_workers`` / ``parallel_backend``
     - Runtime and parallelisation controls.
   * - ``log`` / ``experiment_name``
     - Logging configuration.
   * - ``HPO`` / ``minimization`` / ``_random``
     - Special operation modes.
   * - ``niching`` / ``distance_metric`` / ``niche_radius`` /
       ``adaptive_niche_radius`` / ``clearing_interval``
     - Diversity management.
   * - ``evaluate_population``
     - Use population-level evaluation.
   * - ``diff_mode``
     - Request unified diff patches.
