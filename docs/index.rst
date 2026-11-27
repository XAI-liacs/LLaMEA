:hide-toc:

LLaMEA
======

.. raw:: html

   <div class="llamea-hero">
     <div class="hero-title">LLM-driven <span class="grad">algorithm design</span></div>
     <p>LLaMEA couples large-language-model reasoning with an evolutionary loop
     to invent, mutate and benchmark algorithms fully autonomously. The fully
     open alternative to AlphaEvolve, first released in November 2024.</p>
     <a class="hero-btn primary" href="Quickstart.html">Get started</a>
     <a class="hero-btn ghost" href="variants.html">Variants</a>
     <a class="hero-btn blade" href="blade.html">Benchmark with BLADE &rarr;</a>
   </div>

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Get started
      :link: Quickstart
      :link-type: doc

      Install LLaMEA and evolve a first algorithm in a few lines.

   .. grid-item-card:: :octicon:`git-branch` Variants
      :link: variants
      :link-type: doc

      LLaMEA-HPO, -BO, -MO and -SAGE as ready-made classes.

   .. grid-item-card:: :octicon:`gear` Configure the loop
      :link: llamea
      :link-type: doc

      Strategies, operators, niching, diff mode, warm start.

   .. grid-item-card:: :octicon:`beaker` Benchmark with BLADE
      :link: blade
      :link-type: doc

      Compare LLaMEA with other methods on standard suites.

   .. grid-item-card:: :octicon:`book` Publications
      :link: publications
      :link-type: doc

      Papers on the method, its analysis and applications.

   .. grid-item-card:: :octicon:`code` API reference
      :link: modules
      :link-type: doc

      Every class and function, from ``LLaMEA`` to ``Solution``.

Quick start
-----------

.. tab-set::

   .. tab-item:: LLaMEA

      .. code-block:: python

         from llamea import LLaMEA, OpenAI_LLM

         llm = OpenAI_LLM(api_key="...", model="gpt-4o")

         def evaluate(solution, logger=None):
             namespace = {}
             exec(solution.code, namespace)
             algorithm = namespace[solution.name](budget=1000, dim=5)
             score = ...  # run the algorithm on your problem
             solution.set_scores(score, feedback=f"Score: {score:.3f}")
             return solution

         best = LLaMEA(evaluate, llm, budget=100).run()

   .. tab-item:: A variant

      .. code-block:: python

         from llamea import LLaMEA_BO

         # (4+16)-ES, crossover 0.6, BO task prompt, 100 candidates
         best = LLaMEA_BO(evaluate, llm).run()

   .. tab-item:: In BLADE

      .. code-block:: bash

         pip install iohblade
         iohblade init my-exp
         iohblade run my-exp/experiment.yml   # LLaMEA vs. other methods

At a glance
-----------

.. raw:: html

   <div class="llamea-stats">
     <div class="llamea-stat"><b>5</b><span>ready-made variants from our papers</span></div>
     <div class="llamea-stat"><b>8+</b><span>LLM back-ends, hosted or local</span></div>
     <div class="llamea-stat"><b>15</b><span>papers on method, analysis and applications</span></div>
     <div class="llamea-stat"><b>🥈</b><span>Humies 2025 Silver at GECCO</span></div>
   </div>

.. raw:: html

   <div class="blade-callout">
   <b>LLaMEA designs algorithms; BLADE benchmarks them.</b>
   Use LLaMEA directly for your own problem. To compare it (or a variant) with
   EoH, ReEvo, FunSearch, MCTS-AHD and others on standard problem suites, with
   logging, plots and a web app, use
   <a href="https://github.com/XAI-liacs/BLADE">BLADE</a>
   (<a href="blade.html">how</a>).
   </div>

Cite LLaMEA
-----------

If you use LLaMEA in your research, please cite the
`IEEE TEVC paper <https://doi.org/10.1109/TEVC.2024.3497793>`_ (BibTeX on the
:doc:`publications` page).

.. toctree::
   :hidden:
   :caption: Get started

   Introduction
   Installation
   Quickstart
   notebooks/simple_example

.. toctree::
   :hidden:
   :caption: User guide

   llamea
   variants
   crossover
   llms
   blade

.. toctree::
   :hidden:
   :caption: Examples

   examples
   notebooks/automl_example

.. toctree::
   :hidden:
   :caption: Research

   publications

.. toctree::
   :hidden:
   :caption: Reference

   modules
   changelog
