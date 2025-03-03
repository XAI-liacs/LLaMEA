.. LLaMEA documentation master file, created by
   sphinx-quickstart on Mon Mar  3 09:42:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


LLaMEA: Large Language Model Evolutionary Algorithm
===================================================

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
.. image:: https://codecov.io/gh/nikivanstein/LLaMEA/graph/badge.svg?token=VKCNPWVBNM
   :target: https://codecov.io/gh/nikivanstein/LLaMEA
   :alt: Codecov


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

- 2025.03 ✨✨ **LLaMEA v1.0.0 released**!
- 2025.01 ✨✨ **LLaMEA paper accepted in IEEE TEVC**:
  `Llamea: A large language model evolutionary algorithm for automatically generating metaheuristics <https://ieeexplore.ieee.org/abstract/document/10752628/>`_!



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
     title={Llamea: A large language model evolutionary algorithm for automatically generating metaheuristics},
     author={van Stein, Niki and B{\"a}ck, Thomas},
     journal={IEEE Transactions on Evolutionary Computation},
     year={2024},
     publisher={IEEE}
   }

If you only want to cite the LLaMEA-HPO variant, use the following:

.. code-block:: bibtex

   @article{van2024loop,
     title={In-the-loop hyper-parameter optimization for llm-based automated design of heuristics},
     author={van Stein, Niki and Vermetten, Diederick and B{\"a}ck, Thomas},
     journal={arXiv preprint arXiv:2410.16309},
     year={2024}
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction
   Installation
   Quickstart
   notebooks/simple_example

.. toctree::
   :maxdepth: 2
   :caption: LLaMEA Modules

   llamea.individual
   llamea.llamea
   llamea.llm
   llamea.loggers
   llamea.utils

.. automodule:: llamea.individual
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: llamea.llamea
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: llamea.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: llamea.loggers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: llamea.utils
   :members:
   :undoc-members:
   :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
