
<h1 align="center">LLM Driven Design of Diverse Optimization Problems with High-level Characteristics</h1>

This is the repository for the paper "LLM Driven Design of Diverse Optimization Problems with High-level Characteristics" based on the open-source LLaMEA framework.

LLaMEA couples large-language-model reasoning with an evolutionary loop to **invent, mutate and benchmark algorithms fully autonomously**.  

In this work we extended LLaMEA with a niching approach for the automatic evolution of diverse black-box continuous problem landscapes.

All generated problems and their 2d visualisations can be found in the `landscapes` folder.


## 🎁 Installation

Install the package from source using <a href="https://docs.astral.sh/uv/" target="_blank">uv</a> (0.7.19).
make sure you have `uv` installed.

1. Clone or download the repository:
   ```bash
   cd LLaMEA
   ```
2. Install the required dependencies via uv:
   ```bash
   uv sync
   ```
3. Optional install dev or/and example dependencies:
   ```bash
   uv sync --dev --group examples
   ```

## 💻 Quick Start


1. Set up an OpenAI API key:
   - Obtain an API key from [OpenAI](https://openai.com/).
   - Set the API key in your environment variables:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

2. Running an Experiment

    To run an optimization experiment using LLaMEA:

    ```python
    from llamea import LLaMEA

    # Define your evaluation function
    def your_evaluation_function(solution):
        # Implementation of your function
        # return feedback, quality score, error information
        return "feedback for LLM", 0.1, ""

    # Initialize LLaMEA with your API key and other parameters
    optimizer = LLaMEA(f=your_evaluation_function, api_key="your_api_key_here")

    # Run the optimizer
    best_solution, best_fitness = optimizer.run()
    print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
    ```



## 🤖 Reproducability

The experiments of the paper are in the `ELA.py` file. Additional visualisations can be found in the `ela.ipynb` notebook.
For the verification of the generated landscapes we used the `basinsattribution` method. The main file for post-processing is `basinsattribution-main/process_basins.py`, an example notebook can be found in the notebook: `basinsattribution-main/basins.ipynb`.

