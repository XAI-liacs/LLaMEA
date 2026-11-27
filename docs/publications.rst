Publications
============

LLaMEA is developed by the XAI group of `NACO <https://naco.liacs.nl/>`_ at
LIACS, Leiden University (`project page
<https://naco.liacs.nl/projects/2024-llamea/>`_, `video
<https://www.youtube.com/watch?v=LtFc8K5Uc4w>`_).

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: :octicon:`light-bulb` Methodology
      :link: methodology
      :link-type: ref

      The algorithm and its variants.

   .. grid-item-card:: :octicon:`graph` Benchmarking & analysis
      :link: benchmarking
      :link-type: ref

      BLADE, code evolution graphs, behaviour spaces.

   .. grid-item-card:: :octicon:`rocket` Applications
      :link: applications
      :link-type: ref

      Photonics, real-world optimisation, AutoML.

.. _methodology:

Methodology
-----------

.. rst-class:: pub-list

- **LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically
  Generating Metaheuristics.** *IEEE Transactions on Evolutionary
  Computation* 29(2), 2025.
  `DOI <https://doi.org/10.1109/TEVC.2024.3497793>`__
- **In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design
  of Heuristics** (:class:`~llamea.variants.LLaMEA_HPO`). *ACM Transactions on
  Evolutionary Learning and Optimization*, 2025.
  `DOI <https://doi.org/10.1145/3731567>`__
- **Controlling the Mutation in Large Language Models for the Efficient
  Evolution of Algorithms** (``adaptive_mutation``). *EvoApplications*,
  Springer, 2025.
  `DOI <https://doi.org/10.1007/978-3-031-90065-5_25>`__
- **LLaMEA-BO: A Large Language Model Evolutionary Algorithm for
  Automatically Generating Bayesian Optimization Algorithms**
  (:class:`~llamea.variants.LLaMEA_BO`).
  `arXiv:2505.21034 <https://arxiv.org/abs/2505.21034>`__
- **LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback
  from Explainable AI** (:class:`~llamea.variants.LLaMEA_SAGE`).
  `arXiv:2601.21511 <https://arxiv.org/abs/2601.21511>`__
- **LLM-Driven Evolutionary Generation of Multi-Objective Bayesian
  Optimization Algorithms.**
  `arXiv:2607.08791 <https://arxiv.org/abs/2607.08791>`__
- **Large Language Models and Evolutionary Computation: A Critical Review of
  Bidirectional Interaction, Automated Algorithm Design, and Co-adaptive
  Systems.** *Computer Science Review*, 2026.
  `Article <https://www.sciencedirect.com/science/article/pii/S1574013726001589>`__

.. _benchmarking:

Benchmarking & analysis
-----------------------

.. rst-class:: pub-list

- **BLADE: Benchmark Suite for LLM-driven Automated Design and Evolution of
  Iterative Optimisation Heuristics.** *GECCO '25 Companion*.
  `DOI <https://doi.org/10.1145/3712255.3734347>`__ ·
  `code <https://github.com/XAI-liacs/BLADE>`__ (see :doc:`blade`)
- **Code Evolution Graphs: Understanding Large Language Model Driven Design
  of Algorithms.** *GECCO '25*.
  `DOI <https://doi.org/10.1145/3712256.3726328>`__
- **Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery.**
  `arXiv:2507.03605 <https://arxiv.org/abs/2507.03605>`__
- **From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from
  Strong Priors.** `arXiv:2603.02792 <https://arxiv.org/abs/2603.02792>`__

.. _applications:

Applications
------------

.. rst-class:: pub-list

- **Optimizing Photonic Structures with Large Language Model Driven Algorithm
  Discovery.** *GECCO '25 Companion*.
  `DOI <https://doi.org/10.1145/3712255.3734288>`__
- **Landscape-aware Automated Algorithm Design: An Efficient Framework for
  Real-world Optimization.**
  `arXiv:2602.04529 <https://arxiv.org/abs/2602.04529>`__
- **LLM Driven Design of Continuous Optimization Problems with Controllable
  High-level Properties.**
  `arXiv:2601.18846 <https://arxiv.org/abs/2601.18846>`__
- **Evolving Executable Pipeline Programs for AutoML with Language Models.**
  `arXiv:2608.16416 <https://arxiv.org/abs/2608.16416>`__

Awards
------

- 🥈 Silver award, `Humies @ GECCO 2025 <https://gecco-2025.sigevo.org/Humies-Awards>`_
- 🏆 Winner, `Anytime Performance for Affine BBOB competition, GECCO 2025
  <https://gecco-2025.sigevo.org/Competition-Awards#AA4AffineBBOB_Competition>`_
- 🏆 Winner, `Anytime Algorithms for Many-affine BBOB Functions, GECCO 2024
  <https://gecco-2024.sigevo.org/Competition-Awards.html#Anytime_Algorithms_for_Many-affine_BBOB_Functions>`_
- 🥉 Bronze, `IEEE CoG 2025 Planet Wars RTS competition
  <https://simonlucas.github.io/planet-wars-rts/competitions/IEEE_CoG_2025_Results.html>`_

Reproducibility
---------------

- BLADE – code and results: `doi:10.5281/zenodo.15119985 <https://doi.org/10.5281/zenodo.15119985>`_
- LLaMEA-HPO – code, generated algorithms and IOH logging data: `doi:10.5281/zenodo.14917719 <https://doi.org/10.5281/zenodo.14917719>`_
- Code Evolution Graphs – figures and code: `doi:10.5281/zenodo.14770672 <https://doi.org/10.5281/zenodo.14770672>`_
- LLaMEA: `doi:10.5281/zenodo.13842144 <https://doi.org/10.5281/zenodo.13842144>`_

Cite LLaMEA
-----------

.. code-block:: bibtex

   @article{van2025llamea,
     author  = {van Stein, Niki and B{\"a}ck, Thomas},
     journal = {IEEE Transactions on Evolutionary Computation},
     title   = {LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics},
     year    = {2025},
     volume  = {29},
     number  = {2},
     pages   = {331--345},
     doi     = {10.1109/TEVC.2024.3497793}
   }

LLaMEA-HPO:

.. code-block:: bibtex

   @article{van2025intheloop,
     author  = {van Stein, Niki and Vermetten, Diederick and B{\"a}ck, Thomas},
     title   = {In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics},
     journal = {ACM Transactions on Evolutionary Learning and Optimization},
     year    = {2025},
     doi     = {10.1145/3731567}
   }

Mutation control:

.. code-block:: bibtex

   @inproceedings{yin2025controlling,
     author    = {Yin, Haoran and Kononova, Anna V. and B{\"a}ck, Thomas and van Stein, Niki},
     title     = {Controlling the Mutation in Large Language Models for the Efficient Evolution of Algorithms},
     booktitle = {Applications of Evolutionary Computation},
     publisher = {Springer Nature Switzerland},
     year      = {2025},
     pages     = {403--417},
     doi       = {10.1007/978-3-031-90065-5_25}
   }
