Quick start
===========

.. tip::
   Try the getting-started notebook on
   `Colab <https://colab.research.google.com/github/XAI-liacs/LLaMEA/blob/main/docs/notebooks/simple_example.ipynb>`_.

1. Choose an LLM
----------------

.. code-block:: python

   from llamea import OpenAI_LLM, Gemini_LLM, Ollama_LLM

   llm = OpenAI_LLM(api_key="...", model="gpt-6.2-sol")
   # llm = Gemini_LLM(api_key="...", model="gemini-3.8-flash")
   # llm = Ollama_LLM("qwen2.5-coder:7b")        # local, no key

See :doc:`llms` for all back-ends.

2. Write an evaluation function
-------------------------------

The function receives a :class:`~llamea.solution.Solution` (``code``,
``name``, ``description``) and must set its score and feedback:

.. code-block:: python

   def evaluate(solution, logger=None):
       try:
           namespace = {}
           exec(solution.code, namespace)
           algorithm = namespace[solution.name](budget=1000, dim=5)
           score = run_on_my_problem(algorithm)        # higher is better
           solution.set_scores(score, feedback=f"The algorithm scored {score:.3f}.")
       except Exception as e:
           # The error (with the offending line) is shown to the LLM.
           solution.set_scores(float("-inf"), feedback="The code failed.", error=e)
       return solution

3. Run LLaMEA
-------------

.. code-block:: python

   from llamea import LLaMEA

   optimizer = LLaMEA(
       evaluate,
       llm,
       n_parents=4,
       n_offspring=8,
       budget=100,                # candidates to generate and evaluate
       task_prompt="Write a Python class ... that ...",
   )
   best = optimizer.run()
   print(best.name, best.fitness)
   print(best.code)

LLaMEA maximises by default; pass ``minimization=True`` otherwise. Logs
(conversation, every candidate, the archive for warm start) are written to
``exp-<date>-LLaMEA-<model>-<experiment_name>/``; pass ``log=False`` to
disable them.

Next steps
----------

* Use a ready-made configuration from a paper: :doc:`variants`.
* Tune strategies, operators and diversity: :doc:`llamea` and :doc:`crossover`.
* Compare against other methods on standard problems: :doc:`blade`.
* Browse complete scripts: :doc:`examples`.
