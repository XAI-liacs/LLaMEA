.. LLaMEA documentation master file, created by
   sphinx-quickstart on Mon Mar  3 09:42:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


LLaMEA: Large Language Model Evolutionary Algorithm
===================================================

.. note::
   **The fully-open alternative to Google DeepMind's AlphaEvolve.
   First released 📅 Nov 2024 • MIT License • 100 % reproducible.**

.. note::
   **🥈 Winner of the Silver `Humies 2025 <https://www.human-competitive.org/awards>`_ at GECCO!**

LLaMEA couples large-language-model reasoning with an evolutionary loop to **invent, mutate and benchmark algorithms fully autonomously**.  

.. image:: https://badge.fury.io/py/llamea.svg
   :target: https://pypi.org/project/llamea/
   :alt: PyPI version
   :height: 18
.. image:: https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg
   :alt: Maintenance
   :height: 18
.. image:: https://img.shields.io/badge/Python-3.10+-blue.svg
   :alt: Python 3.10+
   :height: 18
.. image:: https://codecov.io/gh/XAI-liacs/LLaMEA/graph/badge.svg?token=VKCNPWVBNM
   :target: https://codecov.io/gh/XAI-liacs/LLaMEA
   :alt: Codecov
   :height: 18
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13842144.svg
   :target: https://doi.org/10.5281/zenodo.13842144
   :alt: DOI
   :height: 18


**LLaMEA** (Large Language Model Evolutionary Algorithm) is an innovative framework
that leverages the power of large language models (LLMs) such as GPT-4 for the
automated generation and refinement of metaheuristic optimization algorithms.
The framework utilizes a novel approach to evolve and optimize algorithms
iteratively based on performance metrics and runtime evaluations without
requiring extensive prior algorithmic knowledge. This makes LLaMEA an ideal tool
for both research and practical applications in fields where optimization is
crucial.

🔥 News
----
- 2025.07 🎉🎉 **"LLaMEA" won the Any-time Performancy on Many-Affine BBOB competition, and the Silver award at the `Humies @GECCO2025 <https://www.human-competitive.org/awards>`_**!

- 2025.06 🎉🎉 **`"LLaMEA-BO: A Large Language Model Evolutionary Algorithm for Automatically Generating Bayesian Optimization Algorithms" <https://arxiv.org/abs/2505.21034>`_ published on Arxiv**!

- 2025.05 🎉🎉 **`"Optimizing Photonic Structures with Large Language Model Driven Algorithm Discovery" <https://arxiv.org/abs/2503.19742>`_ accepted as workshop paper at GECCO 2025**!

- 2025.05 🎉🎉 **`"BLADE: Benchmark Suite for LLM-Driven Automated Design and Evolution of iterative optimisation heuristics" <https://arxiv.org/abs/2504.20183>`_ accepted as workshop paper at GECCO 2025**!

- 2025.04 🎉🎉 **LLaMEA-HPO paper accepted in ACM TELO** `“In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics" <https://dl.acm.org/doi/abs/10.1145/3731567>`_!

- 2025.04 🎉🎉 **`"Code Evolution Graphs" <https://arxiv.org/abs/2503.16668>`_ accepted as full paper at GECCO 2025**!

- 2025.03 🎉🎉 **LLaMEA v1.0.0 released**!

- 2025.01 🎉🎉 **LLaMEA paper accepted in IEEE TEVC** `“Llamea: A large language model evolutionary algorithm for automatically generating metaheuristics" <https://ieeexplore.ieee.org/abstract/document/10752628/>`_!


🤖 Contributing
------------

Contributions to LLaMEA are welcome! Here are a few ways you can help:

- **Report Bugs**: Use `GitHub Issues <https://github.com/nikivanstein/LLaMEA/issues>`_ to report bugs.
- **Feature Requests**: Suggest new features or improvements.
- **Pull Requests**: Submit PRs for bug fixes or feature additions.

Please refer to ``CONTRIBUTING.md`` for more details on contributing guidelines.

License
-------

Distributed under the `MIT <https://choosealicense.com/licenses/mit/>`_ License.
See ``LICENSE`` for more information.

Cite us
--------

If you use LLaMEA in your research, please consider citing the associated paper:

.. code-block:: bibtex

   @article{van2024llamea,
      author={Stein, Niki van and Bäck, Thomas},
      journal={IEEE Transactions on Evolutionary Computation}, 
      title={LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics}, 
      year={2025},
      volume={29},
      number={2},
      pages={331-345},
      keywords={Benchmark testing;Evolutionary computation;Metaheuristics;Codes;Large language models;Closed box;Heuristic algorithms;Mathematical models;Vectors;Systematics;Automated code generation;evolutionary computation (EC);large language models (LLMs);metaheuristics;optimization},
      doi={10.1109/TEVC.2024.3497793}
   }

If you only want to cite the LLaMEA-HPO variant, use the following:

.. code-block:: bibtex

   @article{van2024loop,
      author = {van Stein, Niki and Vermetten, Diederick and B\"{a}ck, Thomas},
      title = {In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics},
      year = {2025},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3731567},
      doi = {10.1145/3731567},
      note = {Just Accepted},
      journal = {ACM Trans. Evol. Learn. Optim.},
      month = apr,
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction
   Installation
   Quickstart
   notebooks/simple_example
   notebooks/automl_example
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
