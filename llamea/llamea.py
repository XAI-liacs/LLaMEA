"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""

import concurrent.futures
import contextlib
import logging
import os
import random
import re
import traceback
import warnings
from typing import Callable, Optional
import pickle, pickletools
import jsonlines

import numpy as np
from joblib import Parallel, delayed

from .llm import LLM

try:
    from ConfigSpace import ConfigurationSpace
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ConfigurationSpace = None

from .loggers import ExperimentLogger
from .solution import Solution
from .utils import (
    NoCodeException,
    code_distance,
    discrete_power_law_distribution,
    handle_timeout,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        llm,
        n_parents=5,
        n_offspring=5,
        role_prompt="",
        task_prompt="",
        example_prompt=None,
        output_format_prompt=None,
        experiment_name="",
        elitism=True,
        HPO=False,
        mutation_prompts=None,
        adaptive_mutation=False,
        adaptive_prompt=False,
        budget=100,
        eval_timeout=3600,
        max_workers=10,
        parallel_backend="loky",
        log=True,
        minimization=False,
        _random=False,
        niching: Optional[str] = None,
        distance_metric: Optional[Callable[[Solution, Solution], float]] = None,
        niche_radius: Optional[float] = None,
        adaptive_niche_radius: bool = False,
        clearing_interval: Optional[int] = None,
        evaluate_population=False,
        diff_mode: bool = False,
        parent_selection: str = "random",
        tournament_size: int = 3,
    ):
        """
        Initializes the LLaMEA instance with provided parameters. Note that by default LLaMEA maximizes the objective.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            llm (object): An instance of a language model that will be used to generate and evolve algorithms.
            n_parents (int): The number of parents in the population.
            n_offspring (int): The number of offspring each iteration.
            elitism (bool): Flag to decide if elitism (plus strategy) should be used in the evolutionary process or comma strategy.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            example_prompt (str): An example prompt to guide the language model in generating code (or None for default).
            output_format_prompt (str): A prompt that specifies the output format of the language model's response.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            mutation_prompts (list): A list of prompts to specify mutation operators to the LLM model. Each mutation, a random choice from this list is made.
            adaptive_mutation (bool): If set to True, the mutation prompt 'Change X% of the lines of code' will be used in an adaptive control setting.
                This overwrites mutation_prompts.
            adaptive_prompt (bool): If True, the task prompt is optimized before each mutation, allowing it to co-evolve with the individuals.
            budget (int): The number of generations to run the evolutionary algorithm.
            eval_timeout (int): The number of seconds one evaluation can maximum take (to counter infinite loops etc.). Defaults to 1 hour.
            max_workers (int): The maximum number of parallel workers to use for evaluating individuals.
            parallel_backend (str): The backend to use for parallel processing (e.g., 'loky', 'threading').
            log (bool): Flag to switch of the logging of experiments.
            minimization (bool): Whether we minimize or maximize the objective function. Defaults to False.
            _random (bool): Flag to switch to random search (purely for debugging).
            niching (str | None): Niching strategy to use. Supports "sharing" and
                "clearing". If ``None``, niching is disabled.
            distance_metric (callable | None): Function that computes a distance
                between two :class:`Solution` objects. Defaults to a simple AST
                based distance if not supplied.
            niche_radius (float | None): Radius for niche determination when
                using fitness sharing or clearing. If ``None`` a default of ``0.5``
                is used when a niching method is active.
            adaptive_niche_radius (bool): If ``True`` the niche radius adapts to
                the population each generation.
            clearing_interval (int | None): Interval (in generations) at which
                clearing is applied when ``niching`` is set to ``"clearing"``.
            evaluate_population (bool): If True, the evaluation function `f` should
                accept a list with the new population and a list of parents (optionally, to deal with elitism)
                and return a list of solutions that are evaluated, also the parents may receive new fitness values and should be returned.
                So `f` should have the signature
                `f(population, parents=None, logger=None) -> (evaluated_offspring, evaluated_parents)`.
            diff_mode (bool): If ``True``, the LLM is asked to generate unified diff
                patches instead of complete code when evolving solutions.
            parent_selection (str): Strategy for selecting parents to produce
                offspring. Options are ``"random"``, ``"tournament"``, which
                picks the best from a sampled subset, and fitness-proportionate
                roulette wheel selection ``"roulette"``.
            tournament_size (int): Number of candidates sampled in each
                tournament when using ``"tournament"`` selection.
        """
        self.llm = llm
        self.model = llm.model
        self.diff_mode = diff_mode

        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
        self.role_prompt = role_prompt
        self.parallel_backend = parallel_backend
        if HPO and ConfigurationSpace is None:
            warnings.warn(
                "ConfigSpace is not installed. Install ConfigSpace to enable HPO.",
                stacklevel=2,
            )
            HPO = False
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
        if task_prompt == "":
            self.task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Give an excellent and novel heuristic algorithm to solve this task.
"""
        else:
            self.task_prompt = task_prompt

        if example_prompt == None:
            self.example_prompt = """
An example of such code (a simple random search), is as follows:
```
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
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
"""
        else:
            self.example_prompt = example_prompt

        if output_format_prompt is None:
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
        else:
            self.output_format_prompt = output_format_prompt
        self.diff_output_format_prompt = """
Provide only the unified diff patch for the requested changes. Begin with
`--- original.py` and `+++ updated.py` headers and enclose the patch in a
markdown code block labelled as diff:
# Description: <short-description>
```diff
--- original.py
+++ updated.py
@@
<patch>
```
"""
        self.mutation_prompts = mutation_prompts
        self.adaptive_mutation = adaptive_mutation
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",  # small mutation
                # "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
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
        self.evaluate_population = evaluate_population
        self.adaptive_prompt = adaptive_prompt
        self.worst_value = -np.inf
        if minimization:
            self.worst_value = np.inf
        self.niching = niching
        self.distance_metric = distance_metric or code_distance
        self.niche_radius = niche_radius if niche_radius is not None else 0.5
        self.adaptive_niche_radius = adaptive_niche_radius
        self.clearing_interval = clearing_interval
        self.best_so_far = Solution(name="", code="")
        self.best_so_far.set_scores(self.worst_value, "", "")
        self.experiment_name = experiment_name
        self.parent_selection = parent_selection  # "random" | "roulette" | "tournament"
        self.tournament_size = tournament_size

        if self.log:
            modelname = self.model.replace(":", "_")
            self.logger = ExperimentLogger(f"LLaMEA-{modelname}-{experiment_name}")
            self.llm.set_logger(self.logger)
        else:
            self.logger = None
        self.textlog = logging.getLogger(__name__)
        if max_workers > self.n_offspring:
            max_workers = self.n_offspring
        self.max_workers = max_workers
        self.pickle_archive()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("textlog", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Logging object often has open f stream, which is not piklable.
        self.textlog = logging.getLogger(__name__)

    @classmethod
    def warm_start(cls, path_to_archive_dir):
        """
        Class method for warm starts, takes a archive directory, and finds pickle archieve stored at path_to_archieve_dir/llamea_config.pkl,
        generates the object from it, and return it for warm start.
        Args:
            path_to_archive_dir: Directory of instance for which warm start needs to be executed.
        """
        try:
            with open(f"{path_to_archive_dir}/llamea_config.pkl", "rb") as file:
                obj = pickle.load(file)
            return obj
        except Exception as e:
            print(
                f"Error unarchiving object from {path_to_archive_dir}/llamea_config.pkl: {e.__repr__()}"
            )
            return None

    def logevent(self, event):
        self.textlog.info(event)

    def initialize_single(self):
        """
        Initializes a single solution.
        """
        new_individual = Solution(name="", code="", generation=self.generation)
        session_messages = [
            {
                "role": "user",
                "content": self.role_prompt
                + self.task_prompt
                + self.example_prompt
                + self.output_format_prompt,
            },
        ]
        try:
            new_individual = self.llm.sample_solution(session_messages, HPO=self.HPO)
            new_individual.generation = self.generation
            new_individual.task_prompt = self.task_prompt
            if not self.evaluate_population:
                new_individual = self.evaluate_fitness(new_individual)
        except Exception as e:
            new_individual.set_scores(
                self.worst_value,
                f"An exception occured: {traceback.format_exc()}.",
                repr(e) + traceback.format_exc(),
            )
            self.logevent(f"An exception occured: {traceback.format_exc()}.")
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(new_individual)
        print("Releasing individual", new_individual)
        return new_individual

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """

        population = self.population
        population_gen = []
        try:
            timeout = self.eval_timeout
            population_gen = Parallel(
                n_jobs=self.max_workers,
                backend=self.parallel_backend,
                timeout=timeout + 15,
                return_as="generator_unordered",
            )(
                delayed(self.initialize_single)()
                for _ in range(self.n_parents - len(population))
            )
        except Exception as e:
            print(f"Parallel time out in initialization {e}, retrying.")
        for p in population_gen:
            population.append(p)

        if self.evaluate_population:
            population = self.evaluate_population_fitness(population)

        for p in population:
            self.run_history.append(p)

        self.generation += 1
        self.population = population  # Save the entire population
        self.update_best()

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the provided individual by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            individual (Solution): The solution instance to evaluate.

        Returns:
            Solution: The updated solution with feedback, fitness and error information filled in.
        """
        with contextlib.redirect_stdout(None):
            updated_individual = self.f(individual, self.logger)

        return updated_individual

    def evaluate_population_fitness(self, new_population):
        """Evaluate a full population of solutions."""
        with contextlib.redirect_stdout(None):
            # pass the new population and the parent population to the evaluation function
            evaluated_offspring, evaluated_parents = self.f(
                new_population, self.population, self.logger
            )
            self.population = evaluated_parents  # The parent population fitness might also be updated (this does not need to be logged)
        return evaluated_offspring

    def optimize_task_prompt(self, individual):
        """Use the LLM to improve the task prompt for a given individual."""
        prompt = f"""{self.role_prompt}
You are tasked with refining the instructions (task prompt) that guides an LLM to generate algorithms.
### Current task prompt:
----
{individual.task_prompt}
----

### The current algorithm generated with that prompt:
```python
{individual.code}
```

### Feedback from the evaluation on this algorithm:
----
{individual.feedback}
----

Provide an improved / rephrased / augmented task prompt only. The intent of the task prompt should stay the same.
"""
        session_messages = [{"role": "user", "content": prompt}]
        try:
            new_prompt = self.llm.query(session_messages)
            return new_prompt.strip()
        except Exception as e:
            self.logevent(f"Prompt optimization failed: {e}")
            return individual.task_prompt

    def construct_prompt(self, individual):
        """
        Constructs a new session prompt for the language model based on a selected individual.

        Args:
            individual (dict): The individual to mutate.

        Returns:
            list: A list of dictionaries simulating a conversation with the language model for the next evolutionary step.
        """
        # Generate the current population summary
        population_summary = "\n".join([ind.get_summary() for ind in self.population])
        solution = individual.code
        description = individual.description
        feedback = individual.feedback
        if self.adaptive_mutation == True:
            num_lines = len(solution.split("\n"))
            prob = discrete_power_law_distribution(num_lines, 1.5)
            new_mutation_prompt = f"""Refine the strategy of the selected solution to improve it.
Make sure you only change {(prob*100):.1f}% of the code, which means if the code has 100 lines, you can only change {prob*100} lines, and the rest of the lines should remain unchanged.
This input code has {num_lines} lines, so you can only change {max(1, int(prob*num_lines))} lines, the rest {num_lines-max(1, int(prob*num_lines))} lines should remain unchanged.
This changing rate {(prob*100):.1f}% is a mandatory requirement, you cannot change more or less than this rate.
"""
            self.mutation_prompts = [new_mutation_prompt]

        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_operator(mutation_operator)

        task_prompt = (
            individual.task_prompt if self.adaptive_prompt else self.task_prompt
        )
        final_prompt = f"""{task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected solution to update is:
{description}

With code:
{solution}

{feedback}

{mutation_operator}
{self.diff_output_format_prompt if self.diff_mode else self.output_format_prompt}
"""
        session_messages = [
            {"role": "user", "content": self.role_prompt + final_prompt},
        ]

        if self._random:  # not advised to use, only for debugging purposes
            session_messages = [
                {"role": "user", "content": self.role_prompt + self.task_prompt},
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

    def adapt_niche_radius(self, population):
        """Adapt the niche radius based on the current population."""
        if not self.adaptive_niche_radius or len(population) < 2:
            return
        dists = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dists.append(self.distance_metric(population[i], population[j]))
        if dists:
            self.niche_radius = float(np.mean(dists))

    def apply_niching(self, population):
        """Apply the configured niching strategy to ``population``."""
        if self.niching not in {"sharing", "clearing"}:
            return population

        self.adapt_niche_radius(population)

        if self.niching == "sharing":
            for i, ind in enumerate(population):
                niche_count = 1.0
                for j, other in enumerate(population):
                    if i == j:
                        continue
                    d = self.distance_metric(ind, other)
                    if d < self.niche_radius and self.niche_radius > 0:
                        niche_count += 1 - d / self.niche_radius
                if self.minimization:
                    ind.fitness *= niche_count
                else:
                    ind.fitness /= niche_count
        elif self.niching == "clearing":
            if self.clearing_interval and self.generation % self.clearing_interval != 0:
                return population
            reverse = self.minimization == False
            population.sort(key=lambda x: x.fitness, reverse=reverse)
            niches = []
            for ind in population:
                if all(
                    self.distance_metric(ind, winner) >= self.niche_radius
                    for winner in niches
                ):
                    niches.append(ind)
                else:
                    ind.fitness = self.worst_value
        return population

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
        if self.elitism:
            combined_population = parents + offspring
            combined_population = self.apply_niching(combined_population)
            combined_population.sort(key=lambda x: x.fitness, reverse=reverse)
            new_population = combined_population[: self.n_parents]
        else:
            offspring = self.apply_niching(list(offspring))
            offspring.sort(key=lambda x: x.fitness, reverse=reverse)
            new_population = offspring[: self.n_parents]

        return new_population

    def evolve_solution(self, individual):
        """
        Evolves a single solution by constructing a new prompt,
        querying the LLM, and evaluating the fitness.
        """
        individual_copy = individual.copy()
        if self.adaptive_prompt:
            individual_copy.task_prompt = self.optimize_task_prompt(individual_copy)
        new_prompt = self.construct_prompt(individual_copy)

        evolved_individual = individual.empty_copy()
        try:
            evolved_individual = self.llm.sample_solution(
                new_prompt,
                evolved_individual.parent_ids,
                HPO=self.HPO,
                base_code=individual.code,
                diff_mode=self.diff_mode,
            )
            evolved_individual.generation = self.generation
            evolved_individual.task_prompt = individual_copy.task_prompt
            if not self.evaluate_population:
                evolved_individual = self.evaluate_fitness(evolved_individual)
        except Exception as e:
            error = repr(e)
            evolved_individual.generation = self.generation
            evolved_individual.set_scores(
                self.worst_value, f"An exception occurred: {error}.", error
            )
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(evolved_individual)
            self.logevent(f"An exception occured: {traceback.format_exc()}.")

        # self.progress_bar.update(1)
        return evolved_individual

    def get_population_from(self, archive_path):
        """
        Finds population log in archive_path/log.jsonl and loads it to current population.
        If population size in log file is insufficient, runs initialize() for rest of the population.
        Used to run a cold started algorithm with best known population.
        `Note`: Make sure the goal of initialisation of current instance of LLaMEA matches the population being selected.
        Args:
            archive_path: A directory from previous runs, to load well known population from.
        """

        data = []
        try:
            with jsonlines.open(f"{archive_path}/log.jsonl") as reader:
                for obj in reader:
                    data.append(obj)

        except Exception as e:
            self.textlog.error("Error reading population: " + e.__repr__())

        restore_population = data[-1 * self.n_parents :]
        population = []
        print(
            f"Restoring population of size {len(restore_population)}, of {self.n_parents}"
        )
        for individual in restore_population:
            print("\tRestoring...")
            for key, value in individual.items():
                print(f"{key}: {value}, ({type(value)})")
            soln = Solution(
                code=individual["code"],
                name=individual["name"],
                description=individual["description"],
                configspace=(
                    None
                    if individual["configspace"] == ""
                    else individual["configspace"]
                ),
                operator=individual["operator"],
                task_prompt=individual["task_prompt"],
            )
            population.append(soln)

        self.population = population
        if len(population) < self.n_parents:
            print(len(population), self.n_parents)
            self.initialize()
        else:
            print("-----------Init not called--------------")

    def _select_parents(self):
        """
        Return list of parents (length = n_offspring) chosen according to
        self.parent_selection.
        """
        method = (self.parent_selection or "random").lower()
        if method == "random":
            return np.random.choice(self.population, self.n_offspring, replace=True)
        elif method == "roulette":
            return self._roulette_wheel_selection(self.n_offspring)
        elif method == "tournament":
            return self._tournament_selection(self.n_offspring, self.tournament_size)
        else:
            raise ValueError(f"Unknown parent_selection: {self.parent_selection}")

    def _sorted_indices_by_fitness(self):
        """Return indices of population sorted by fitness (best first for maximizing)."""
        reverse = self.minimization == False  # maximize -> reverse True
        return sorted(
            range(len(self.population)),
            key=lambda i: self.population[i].fitness,
            reverse=reverse,
        )

    def _roulette_wheel_selection(self, k: int):
        """
        Rank-based roulette:
        - Convert ordering into weights (best gets largest weight).
        - Works robustly if raw fitness values are negative/inf.
        """
        n = len(self.population)
        if n == 0:
            return []
        sorted_idx = self._sorted_indices_by_fitness()
        # Assign rank weights: best -> n, worst -> 1
        weights = np.zeros(n, dtype=float)
        for pos, idx in enumerate(sorted_idx):
            weights[idx] = n - pos
        s = weights.sum()
        if s <= 0 or not np.isfinite(s):
            probs = np.ones(n) / n
        else:
            probs = weights / s
        chosen_indices = np.random.choice(range(n), size=k, replace=True, p=probs)
        return [self.population[i] for i in chosen_indices]

    def _tournament_selection(self, k: int, tournament_size: int = 3):
        """
        For each parent to choose, sample `tournament_size` candidates randomly
        and pick the best among them.
        """
        n = len(self.population)
        if n == 0:
            return []
        ts = max(1, min(tournament_size, n))
        selected = []
        for _ in range(k):
            if n >= ts:
                cand_idx = random.sample(range(n), ts)
            else:
                cand_idx = list(np.random.choice(range(n), size=ts, replace=True))
            if self.minimization:
                best_i = min(cand_idx, key=lambda i: self.population[i].fitness)
            else:
                best_i = max(cand_idx, key=lambda i: self.population[i].fitness)
            selected.append(self.population[best_i])
        return selected

    def run(self, archive_path=None):
        """
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Args:
            archive_path: Runs the algorithm with a given known population, and performs
        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        if archive_path != None:
            self.logevent(f"Loading population from {archive_path}/log.jsonl...")
            self.get_population_from(archive_path)
        else:
            self.logevent("No archive path provided, standard initialisation.")
            # self.progress_bar = tqdm(total=self.budget)
            self.logevent("Initializing first population")
            self.initialize()  # Initialize a population
            # self.progress_bar.update(self.n_parents)

        if self.log:
            self.logger.log_population(self.population)

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:
            # pick a new offspring population using random sampling
            new_offspring_population = self._select_parents()

            new_population = []
            try:
                timeout = self.eval_timeout
                new_population_gen = Parallel(
                    n_jobs=self.max_workers,
                    timeout=timeout + 15,
                    backend=self.parallel_backend,
                    return_as="generator_unordered",
                )(
                    delayed(self.evolve_solution)(individual)
                    for individual in new_offspring_population
                )
            except Exception as e:
                print("Parallel time out .")

            for p in new_population_gen:
                new_population.append(p)

            if self.evaluate_population:
                new_population = self.evaluate_population_fitness(new_population)

            for p in new_population:
                self.run_history.append(p)

            self.generation += 1

            if self.log:
                self.logger.log_population(new_population)

            # Update population and the best solution
            self.population = self.selection(self.population, new_population)
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )

            ## Archive progress.
            self.pickle_archive()

        return self.best_so_far

    def pickle_archive(self):
        """
        Store the llmea object, into a file, using pickle, to support warm start.
        """
        try:
            with open(f"{self.logger.dirname}/llamea_config.pkl", "wb") as file:
                pickle.dump(self, file)
        except Exception as e:
            self.textlog.error("Error archiving LLaMEA object: " + e.__repr__())
            traceback.print_exc()
