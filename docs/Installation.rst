Installation
============

.. code-block:: bash

   pip install llamea

.. important::
   LLaMEA needs Python **3.11.5 or higher**, and an API key for your LLM
   provider (or a local model through Ollama, LM Studio or MLX).

Optional extras
---------------

.. list-table::
   :header-rows: 1

   * - Install
     - For
   * - ``pip install "llamea[feature-guidance]"``
     - XGBoost for :class:`~llamea.variants.LLaMEA_SAGE` /
       ``feature_guided_mutation`` (~290 MB on Linux)
   * - ``pip install llamea-lite``
     - A slim install without bundled LLM or HPO dependencies (install
       ``ConfigSpace`` yourself for HPO)
   * - ``pip install iohblade``
     - :doc:`BLADE <blade>`, to benchmark LLaMEA against other methods

From source
-----------

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/XAI-liacs/LLaMEA.git
   cd LLaMEA
   uv sync                           # core
   uv sync --dev --group examples    # tests and example dependencies
   uv sync --group docs              # to build this documentation

.. note::
   Some example dependencies need `SWIG <https://www.swig.org/download.html>`_
   (``brew install swig`` / ``sudo apt-get install swig``). The plasmonic
   photonics problem additionally needs ``pyGDM2`` (``uv pip install pyGDM2``),
   which requires ``numpy<=2.3`` and is therefore not part of any group.
