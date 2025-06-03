Quick Start
-----------

1. Set up an OpenAI API key:

   - Obtain an API key from `OpenAI <https://openai.com/>`_. (or other supported LLM providers)

2. Running an Experiment

   .. code-block:: python

      from llamea import LLaMEA, OpenAI_LLM

      llm = OpenAI_LLM(model="gpt-3.5-turbo", api_key="your_api_key_here")

      # Define your evaluation function
      def your_evaluation_function(solution, explogger=None):
          # Implementation of your function
          print(solution.code, solution.name) #the code and name generated.
          # Set fitness and feedback
          solution.set_scores(1.0, "Great solution, with score 1.0")
          return solution

      # Initialize LLaMEA with your API key and other parameters
      optimizer = LLaMEA(f=your_evaluation_function, llm=llm)

      # Run the optimizer
      best_solution = optimizer.run()
      print(f"Best Solution: {best_solution.solution}, Fitness: {best_solution.fitness}")

Examples
--------

Below are three example scripts demonstrating LLaMEA in action for black-box
optimization with a BBOB (24 noiseless) function suite, and one Automated Machine Learning use-case.   
One of the black-box optimization scripts (`example.py`) runs basic LLaMEA, while the other (`example_HPO.py`) incorporates
a **hyper-parameter optimization** pipeline—known as **LLaMEA-HPO**—that employs
SMAC to tune the algorithm’s parameters in the loop.

Running ``example.py``
~~~~~~~~~~~~~~~~~~~~~~

**example.py** showcases a straightforward use-case of LLaMEA. It:

- Defines an evaluation function ``evaluateBBOB`` that runs generated algorithms
  on a standard set of BBOB problems (24 functions).
- Initializes LLaMEA with a specific model (e.g., GPT-4, GPT-3.5) and prompts the
  LLM to generate metaheuristic code.
- Iterates over a (1+1)-style evolutionary loop, refining the code until a certain
  budget is reached.

How to run:

.. code-block:: bash

   python example.py

The script will:

1. Query the specified LLM with a prompt describing the black-box optimization task.
2. Dynamically execute each generated algorithm on BBOB problems.
3. Log performance data such as AOCC (Area Over the Convergence Curve).
4. Iteratively refine the best-so-far algorithms.

Running ``example_HPO.py`` (LLaMEA-HPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**example_HPO.py** extends LLaMEA with **in-the-loop hyper-parameter optimization**—
termed **LLaMEA-HPO**. Instead of having the LLM guess or refine hyper-parameters
directly, the code:

- Allows the LLM to generate a Python class representing the metaheuristic
  **plus** a ConfigSpace dictionary describing hyper-parameters.
- Passes these hyper-parameters to SMAC, which then searches for good parameter
  settings on a BBOB training set.
- Evaluates the best hyper-parameters found by SMAC on the full BBOB suite.
- Feeds back the final performance (and errors) to the LLM, prompting it to
  mutate the algorithm’s structure (rather than simply numeric settings).

Why LLaMEA-HPO?
***************

Offloading hyper-parameter search to SMAC significantly reduces LLM query
overhead and encourages the LLM to focus on novel structural improvements.

How to run:

.. code-block:: bash

   python example_HPO.py

Script outline:

1. Prompt & Generation: Script sets up a role/task prompt, along with hyper-parameter
   config space templates.
2. HPO Step: For each newly generated algorithm, SMAC tries different parameter values
   within a budget.
3. Evaluation: The final best configuration from SMAC is tested across BBOB instances.
4. Refinement: The script returns the performance to LLaMEA, prompting the LLM to
   mutate the algorithm design.

.. note::

   Adjust the model name (``ai_model``) or API key as needed in the script.
   Changing ``budget`` or the HPO budget can drastically affect runtime and cost.
   Additional arguments (e.g., logging directories) can be set if desired.


Running ``example_AutoML.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**`example_AutoML.py`** uses LLaMEA to showcase that it can not only evolve and generate metaheuristics but also all kind of other algorithms, such as Machine Learning pipelines.  
In this example, a basic classification task on the breast-cancer dataset from sklearn is solved by generating and evolving open-ended ML pipelines.

- We define the evaluate function (accuracy score on a hold-out test set)
- We provide a very basic example code to get the algorithm started.
- We run a few iterations and observe the excellent performance of our completely automatic ML pipeline.


**How to run:**

.. code-block:: bash
   python example_AutoML.py

.. note::
   Adjust the model name (`ai_model`) or API key as needed in the script.
   You can easily change the dataset, task and evaluation function to fit your needs.