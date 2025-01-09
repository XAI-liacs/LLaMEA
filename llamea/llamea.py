"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
import re
import traceback
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import concurrent.futures
from ConfigSpace import ConfigurationSpace
import random

from .llm import LLMmanager
from .loggers import ExperimentLogger
from .utils import NoCodeException, handle_timeout
from .individual import Individual
from .fmut_beta import discrete_power_law_distribution


# TODOs:
# Implement diversity selection mechanisms (none, prefer short code, update population only when (distribution of) results is different, AST / code difference)


class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        api_key,
        n_parents=5,
        n_offspring=10,
        role_prompt="",
        task_prompt="",
        experiment_name="",
        elitism=False,
        HPO=False,
        mutation_prompts=None,
        budget=100,
        model="gpt-4-turbo",
        eval_timeout=3600,
        log=True,
        minimization=False,
        _random=False,
    ):
        """
        Initializes the LLaMEA instance with provided parameters. Note that by default LLaMEA maximizes the objective.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            api_key (str): The API key for accessing OpenAI's services.
            n_parents (int): The number of parents in the population.
            n_offspring (int): The number of offspring each iteration.
            elitism (bool): Flag to decide if elitism (plus strategy) should be used in the evolutionary process or comma strategy.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            mutation_prompts (list): A list of prompts to specify mutation operators to the LLM model. Each mutation, a random choice from this list is made.
            budget (int): The number of generations to run the evolutionary algorithm.
            model (str): The model identifier from OpenAI or ollama to be used.
            eval_timeout (int): The number of seconds one evaluation can maximum take (to counter infinite loops etc.). Defaults to 1 hour.
            log (bool): Flag to switch of the logging of experiments.
            minimization (bool): Whether we minimize or maximize the objective function. Defaults to False.
            _random (bool): Flag to switch to random search (purely for debugging).
        """
        self.client = LLMmanager(api_key, model)
        self.api_key = api_key
        self.model = model
        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
        if task_prompt == "":
            self.task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
An example of such code (a simple random search), is as follows:
```
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
Give an excellent and novel heuristic algorithm to solve this task.
"""
        else:
            self.task_prompt = task_prompt

        self.output_format_prompt = """
Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""
        if HPO:
            self.output_format_prompt = """
Provide the Python code, a one-line description with the main idea (without enters) and the SMAC3 Configuration space to optimize the code (in Python dictionary format). Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
Space: <configuration_space>"""
        self.mutation_prompts = mutation_prompts
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",  # small mutation
                "Generate a new algorithm that is different from the solutions you have tried before.", #new random solution
            ]
        self.budget = budget
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.population = []
        self.elitism = elitism
        self.generation = 0
        self.run_history = []
        self.log = log
        self._random = _random
        self.HPO = HPO
        self.minimization = minimization
        self.worst_value = -np.Inf
        if minimization:
            self.worst_value = np.Inf
        self.best_so_far = Individual("", "", "", None, 0, None)
        self.best_so_far.set_scores(self.worst_value, "", "")

        if self.log:
            modelname = self.model.replace(":", "_")
            self.logger = ExperimentLogger(
                f"LLaMEA-{modelname}-{experiment_name}")
        else:
            self.logger = None
        self.textlog = logging.getLogger(__name__)

    def logevent(self, event):
        with logging_redirect_tqdm():
            self.textlog.info(event)

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """

        def initialize_single():
            """
            Initializes a single solution.
            """
            new_individual = Individual(
                "", "", "", None, self.generation, None)
            session_messages = [
                {"role": "system", "content": self.role_prompt},
                {
                    "role": "user",
                    "content": self.task_prompt + self.output_format_prompt,
                },
            ]
            try:
                new_individual = self.llm(session_messages)
                new_individual = self.evaluate_fitness(new_individual)
            except NoCodeException:
                new_individual.set_scores(
                    self.worst_value, "No code was extracted.")
            except Exception as e:
                new_individual.set_scores(
                    self.worst_value,
                    f"An exception occured: {traceback.format_exc()}.",
                    repr(e) + traceback.format_exc(),
                )
                self.textlog.warning(new_individual.error)

            self.run_history.append(new_individual)  # update the history
            self.progress_bar.update(1)
            return new_individual

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            population = list(
                executor.map(lambda _: initialize_single(),
                             range(self.n_parents))
            )

        self.generation += 1
        self.population = population  # Save the entire population
        self.update_best()

    def llm(self, session_messages, parent_id=None):
        """
        Interacts with a language model to generate or mutate solutions based on the provided session messages.

        Args:
            session_messages (list): A list of dictionaries with keys 'role' and 'content' to simulate a conversation with the language model.

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        if self.log:
            self.logger.log_conversation(
                "LLaMEA", "\n".join([d["content"] for d in session_messages])
            )

        message = self.client.chat(session_messages)

        if self.log:
            self.logger.log_conversation(self.model, message)

        code = self.extract_algorithm_code(message)
        name = re.findall(
            "class\\s*(\\w*)(?:\\(\\w*\\))?\\:",
            code,
            re.IGNORECASE,
        )[0]
        desc = self.extract_algorithm_description(message)
        cs = None
        if self.HPO:
            cs = self.extract_configspace(message)
        new_individual = Individual(
            code, name, desc, cs, self.generation, parent_id)

        return new_individual

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the provided individual by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            individual (dict): Including required keys "_solution", "_name", "_description" and optional "_configspace" and others.

        Returns:
            tuple: Updated individual with "_feedback", "_fitness" (float), and "_error" (string) filled.
        """
        # Implement fitness evaluation and error handling logic.
        updated_individual = self.f(individual, self.logger)

        return updated_individual

    def construct_prompt(self, individual):
        """
        Constructs a new session prompt for the language model based on a selected individual.

        Args:
            individual (dict): The individual to mutate.

        Returns:
            list: A list of dictionaries simulating a conversation with the language model for the next evolutionary step.
        """
        # Generate the current population summary
        population_summary = "\n".join(
            [ind.get_summary() for ind in self.population])
        solution = individual.solution
        description = individual.description
        feedback = individual.feedback
        # TODO make a random selection between multiple feedback prompts (mutations)
        num_lines = len(solution.split("\n"))
        print(f"number of lines: {num_lines}")
        prob = discrete_power_law_distribution(num_lines, 1.5)
        # prob = 0.4
        new_mutation_prompt = f"""
Now, refine the strategy of the selected solution to improve it. Make sure you 
only change {(prob*100):.1f}% of the code, which means if the code has 100 lines, you 
can only change {prob*100} lines, and the rest lines should remain the same. For 
this code, it has {num_lines} lines, so you can only change {max(1, int(prob*num_lines))}
lines, the rest {num_lines-max(1, int(prob*num_lines))} lines should remain the same. 
This changing rate {(prob*100):.1f}% is the mandatory requirement, you cannot change 
more or less than this rate.
"""
        self.mutation_prompts += [new_mutation_prompt]
        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_mutation_prompt(mutation_operator)

        final_prompt = f"""{self.task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected solution to update is:
{description}

With code:
{solution}

{feedback}

{mutation_operator}
{self.output_format_prompt}
"""
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": final_prompt},
        ]

        if self._random:  # not advised to use, only for debugging purposes
            session_messages = [
                {"role": "system", "content": self.role_prompt},
                {"role": "user", "content": self.task_prompt},
            ]
        # Logic to construct the new prompt based on current evolutionary state.
        return session_messages

    def update_best(self):
        """
        Update the best individual in the new population
        """
        if self.minimization == False:
            best_individual = max(self.population, key=lambda x: x.fitness)

            if best_individual.fitness > self.best_so_far.fitness:
                self.best_so_far = best_individual
        else:
            best_individual = min(self.population, key=lambda x: x.fitness)

            if best_individual.fitness < self.best_so_far.fitness:
                self.best_so_far = best_individual

    def selection(self, parents, offspring):
        """
        Select the new population based on the parents and the offspring and the current strategy.

        Args:
            parents (list): List of solutions.
            offspring (list): List of new solutions.

        Returns:
            list: List of new selected population.
        """
        reverse = self.minimization == False

        # TODO filter out non-diverse solutions
        if self.elitism:
            # Combine parents and offspring
            combined_population = parents + offspring
            # Sort by fitness
            combined_population.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals to form the new population
            new_population = combined_population[: self.n_parents]
        else:
            # Sort offspring by fitness
            offspring.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals from offspring to form the new population
            new_population = offspring[: self.n_parents]

        return new_population

    def extract_configspace(self, message):
        """
        Extracts the configuration space definition in json from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            ConfigSpace: Extracted configuration space object.
        """
        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                c = ConfigurationSpace(eval(m.group(1)))
            except Exception as e:
                self.textlog.warning(
                    "Could not extract configuration space", e.with_traceback
                )
                pass
        return c

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            self.textlog.warning("Message contained no code block")
            raise NoCodeException

    def extract_algorithm_description(self, message):
        """
        Extracts algorithm description from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"#\s*Description\s*:\s*(.*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""

    def run(self, max_workers=10):
        """
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        self.progress_bar = tqdm(total=self.budget)
        self.initialize()  # Initialize a population

        if self.log:
            self.logger.log_population(self.population)

        def evolve_solution(individual):
            """
            Evolves a single solution by constructing a new prompt,
            querying the LLM, and evaluating the fitness.
            """
            new_prompt = self.construct_prompt(individual)
            evolved_individual = individual.copy()

            try:
                evolved_individual = self.llm(
                    new_prompt, evolved_individual.parent_id)
                evolved_individual = self.evaluate_fitness(evolved_individual)
            except NoCodeException:
                evolved_individual.set_scores(
                    self.worst_value,
                    "No code was extracted. The code should be encapsulated with ``` in your response.",
                    "The code should be encapsulated with ``` in your response.",
                )
            except Exception as e:
                error = repr(e)
                evolved_individual.set_scores(
                    self.worst_value, f"An exception occurred: {error}.", error
                )

            self.run_history.append(evolved_individual)
            self.progress_bar.update(1)
            return evolved_individual

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:
            # pick a new offspring population using random sampling
            new_offspring_population = np.random.choice(
                self.population, self.n_offspring, replace=True
            )

            new_population = []
            # Use ThreadPoolExecutor for parallel evolution of solutions
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # new_population = list(executor.map(evolve_solution, new_offspring_population))

                future_to_offspring = {
                    executor.submit(evolve_solution, offspring): offspring
                    for offspring in new_offspring_population
                }
                try:
                    for future in concurrent.futures.as_completed(
                        future_to_offspring,
                        timeout=self.eval_timeout *
                            (max_workers / max_workers),
                    ):
                        try:
                            result = future.result(
                                timeout=self.eval_timeout
                            )  # Apply timeout for each future result
                            new_population.append(result)
                        except concurrent.futures.TimeoutError:
                            self.textlog.warning(
                                "Timeout occurred for one of the solutions"
                            )
                except concurrent.futures.TimeoutError:
                    self.textlog.warning(
                        "Timeout occurred for the as_completed event, pop count:" +
                        str(len(new_population))
                    )
                    executor.shutdown(False)  # stop tasks that took too long

            self.generation += 1

            if self.log:
                self.logger.log_population(new_population)

            # Update population and the best solution
            self.population = self.selection(self.population, new_population)
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )
        self.progress_bar.close()
        return self.best_so_far
