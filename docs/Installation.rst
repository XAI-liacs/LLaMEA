Installation
------------
.. note:: SWIG Dependency

   Ensure that **SWIG** is installed on your system by running:

   .. code-block:: bash

      swig --version

   If it is not installed, use the appropriate method for your platform:

   - **macOS**:

     .. code-block:: bash

        brew install swig

   - **Ubuntu/Linux**:

     .. code-block:: bash

        sudo apt-get install swig
        (or use your distribution's package manager)

   - **Windows**: Download and install from the `official SWIG website <https://www.swig.org/download.html>`_ and make sure to add it to your system PATH.

It is easiest to use LLaMEA from the PyPI package:

.. code-block:: bash

   pip install llamea

.. important::
   The Python version **must** be >= 3.11.5.
   An OpenAI/Gemini/Ollama API key is needed for using LLM models.

You can also install the package from source using uv (0.7.19).
make sure you have **uv** installed.

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/xai-liacs/LLaMEA.git
      cd LLaMEA

2. Install the required dependencies via uv:

   .. code-block:: bash

      uv sync

3. Optionally install dev or/and example dependencies:

   .. code-block:: bash

      uv sync --dev --group examples

   The plasmonic nanostructure problem in
   ``benchmarks/tuto_global_optimization_photonics`` additionally needs
   `pyGDM2 <https://pypi.org/project/pyGDM2/>`_, which is not part of any group
   because it requires ``numpy<=2.3``. Install it separately if you need that
   problem:

   .. code-block:: bash

      uv pip install pyGDM2
