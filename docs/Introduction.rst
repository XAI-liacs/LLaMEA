
Introduction
------------

**LLaMEA** (Large Language Model Evolutionary Algorithm) is an innovative framework
that leverages the power of large language models (LLMs) such as GPT-4 for the
automated generation and refinement of metaheuristic optimization algorithms.
The framework utilizes a novel approach to evolve and optimize algorithms
iteratively based on performance metrics and runtime evaluations without
requiring extensive prior algorithmic knowledge. This makes LLaMEA an ideal tool
for both research and practical applications in fields where optimization is
crucial.

**Key Features:**

- **Automated Algorithm Generation**: Automatically generates and refines
  algorithms using GPT-based or similar LLM models.
- **Performance Evaluation**: Integrates seamlessly with the IOHexperimenter for
  real-time performance feedback, guiding the evolutionary process.
- **LLaMEA-HPO**: Provides an in-the-loop hyper-parameter optimization mechanism
  (via SMAC) to offload numerical tuning, so that LLM queries focus on novel
  structural improvements.
- **Extensible & Modular**: You can easily integrate additional models and
  evaluation tools.

.. image:: framework.png
   :align: center
   :alt: LLaMEA framework
   :width: 100%