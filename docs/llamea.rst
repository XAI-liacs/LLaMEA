LLaMEA
======

The :class:`~llamea.llamea.LLaMEA` class implements the evolutionary loop
around a large language model. Its behaviour is governed by many
hyper-parameters controlling population sizes, mutation style, diversity and
evaluation.

Recent features include:

* **Niching** – enable ``niching="sharing"`` or ``niching="clearing"`` to
  maintain diversity. ``distance_metric``, ``niche_radius``,
  ``adaptive_niche_radius`` and ``clearing_interval`` further tune the niches.
* **Unified diff mode** – set ``diff_mode=True`` to request unified diff patches
  instead of entire source files from the LLM.
* **Population evaluation** – with ``evaluate_population=True`` the evaluation
  function ``f`` operates on lists of solutions, allowing batch evaluations.
* **Warm start** -With every iteration, **LLaMEA** archives its latest run in
    `<experiment_log_directory>/llamea_config.pkl`. The framework provides
    **``warm_start`` class methods** that allow you to resume from a previously
    saved state. This methods:
        - Accept the path to the ``<experiment_log_directory>``.
        - Restore the most recent object from the archive.
        - Reinitialize the program in warm-start mode.
    After restoring the object, you can call `restored_object.run()` to continue execution
    from the point where the program was last terminated, while updating the same experiment
    directory.
*  **Initial Population**  -After a cold start, initialisation of LLaMEA object
   one can use `.run(<experiment_log_directory>)` to start with latest individual
   from the run logged in that directory. Make sure to use similar initialisation
   criteria, as was used in the logged experiment.

Initialization Parameters
-------------------------

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
   * - ``mutation_prompts`` / ``adaptive_mutation`` / ``adaptive_prompt``
     - Mutation and prompt adaptation settings.
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

.. automodule:: llamea.llamea
   :members:
   :undoc-members:
   :show-inheritance:
