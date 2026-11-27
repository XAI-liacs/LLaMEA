Benchmarking with BLADE
=======================

.. raw:: html

   <div class="blade-callout">
   <b>LLaMEA designs algorithms; BLADE benchmarks them.</b>
   Use LLaMEA directly for your own problem and evaluation function. Use
   <a href="https://github.com/XAI-liacs/BLADE">BLADE</a> to compare LLaMEA,
   or one of its variants, with other LLM-driven design methods and report
   the results.
   </div>

`BLADE <https://github.com/XAI-liacs/BLADE>`_ (Benchmarking LLM-driven
Automated Design and Evolution, `GECCO '25
<https://doi.org/10.1145/3712255.3734347>`_) is our benchmarking platform for
automated algorithm design. It provides:

* **Methods**: LLaMEA next to EoH, ReEvo, FunSearch, ShinkaEvolve, MCTS-AHD,
  LHNS and random search, all with the same budget.
* **Problems**: BBOB, SBOX-COST, MA-BBOB, photonics, AutoML, kernel tuning,
  AlphaEvolve-style geometry problems and more, with human-designed baselines.
* **Fair, isolated evaluation**: every candidate runs in its own process and
  virtual environment, with seeding and timeouts.
* **Logging and analysis**: convergence plots, code-evolution graphs,
  behaviour metrics and a results web app.
* **Experiments as configuration**: one YAML file, checked before any LLM
  call is made.

Which one should I use?
-----------------------

.. list-table::
   :header-rows: 1

   * - You want to ...
     - Use
   * - evolve an algorithm for *your* problem, with your own evaluation code
     - LLaMEA (this package)
   * - compare LLaMEA settings or variants on standard problems
     - BLADE
   * - compare LLaMEA with other LLM-driven methods, with plots and tables for a paper
     - BLADE
   * - develop a new LLM-driven design method and benchmark it
     - BLADE (custom method)

A first experiment
------------------

.. code-block:: bash

   pip install iohblade
   iohblade init my-exp                       # starter experiment.yml
   iohblade validate my-exp/experiment.yml    # every mistake, with its line number
   iohblade check my-exp/experiment.yml       # smoke test, no LLM calls
   iohblade run my-exp/experiment.yml         # run (rerun to resume)

LLaMEA's arguments go under ``params``; this runs plain LLaMEA, the
LLaMEA-HPO configuration and random search on SBOX-COST:

.. code-block:: yaml

   experiment: {output_dir: results/my-exp, runs: 5, budget: 100}
   llms:
     gemini: {provider: gemini, model: gemini-2.0-flash, api_key: "${env:GEMINI_API_KEY}"}
   methods:
     - {type: llamea, name: "LLaMEA-{llm}", llm: gemini}
     - type: llamea
       name: "LLaMEA-HPO-{llm}"
       llm: gemini
       params: {HPO: true, n_parents: 1, n_offspring: 1}
     - {type: random_search, name: "RS-{llm}", llm: gemini}
   problems:
     - {type: bbob_sbox, name: SBOX-COST}

The same experiment with BLADE's Python API:

.. code-block:: python

   from iohblade.benchmarks import BBOB_SBOX
   from iohblade.experiment import Experiment
   from iohblade.llm import Gemini_LLM
   from iohblade.loggers import ExperimentLogger
   from iohblade.methods import LLaMEA, RandomSearch

   llm = Gemini_LLM(api_key, "gemini-2.0-flash")
   methods = [
       LLaMEA(llm, budget=100, name="LLaMEA"),
       LLaMEA(llm, budget=100, name="LLaMEA-HPO", HPO=True, n_parents=1, n_offspring=1),
       RandomSearch(llm, budget=100),
   ]
   experiment = Experiment(methods=methods, problems=[BBOB_SBOX()], runs=5,
                           exp_logger=ExperimentLogger("results/my-exp"))
   experiment()

The terminal hint
-----------------

When LLaMEA runs on its own it prints, once per process, a one-line pointer
to BLADE. It is not shown when LLaMEA runs inside BLADE. Set the environment
variable ``LLAMEA_NO_BLADE_HINT=1`` to hide it.

More
----

* BLADE documentation: https://xai-liacs.github.io/BLADE/
* BLADE on GitHub: https://github.com/XAI-liacs/BLADE
* BLADE on PyPI: https://pypi.org/project/iohblade/
