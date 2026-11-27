Examples
========

The ``examples/`` directory holds complete scripts. Set the model and API key
in the script, then run it with ``uv run python examples/<script>.py``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Shows
   * - ``minimum_example.py``
     - The smallest complete set-up.
   * - ``black-box-optimization.py``
     - Metaheuristics for the 24 noiseless BBOB functions, scored by AOCC
       (area over the convergence curve).
   * - ``black-box-opt-with-HPO.py``
     - **LLaMEA-HPO**: the LLM also writes a ConfigSpace and SMAC tunes each
       algorithm on training instances before it is scored.
   * - ``black-box-optimization-diff-mode.py``
     - Diff mode: the LLM answers with SEARCH/REPLACE patches.
   * - ``crossover.py``
     - Mutation and crossover operators, tournament selection and
       adaptive operator weights with discounted UCB.
   * - ``multi_objective.py``
     - **LLaMEA-MO** on a vehicle-routing problem with two objectives
       (distance and fuel); returns the Pareto front.
   * - ``automl_example.py``
     - Evolving ML pipelines (breast-cancer classification) instead of
       optimisers.
   * - ``warm-start-tests.py``
     - Resuming a run from its archive.

The ``benchmarks/`` directory holds the set-ups of our papers (MA-BBOB,
photonic structures, mini-Bragg mirrors).

Viewing conversation logs
-------------------------

``logreader/app.py`` is a small Flask app to browse a run's LLM
conversation:

.. code-block:: bash

   uv run python logreader/app.py --logfile path/to/conversationlog.jsonl

Then open http://localhost:5001.
