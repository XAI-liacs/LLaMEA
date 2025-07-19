Installation
------------

It is easiest to use LLaMEA from the PyPI package:

.. code-block:: bash

   pip install llamea

.. important::
   The Python version **must** be >= 3.11.
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
