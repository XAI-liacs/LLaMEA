import unittest
import os
import time

import numpy as np
import jsonlines

from llamea import LLaMEA, Dummy_LLM
from misc.utils import aoc_logger, OverBudgetException, correct_aoc


def evaluateBBOB(solution, explogger=None):
    auc_mean = 0
    auc_std = 0

    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())

    error = ""

    aucs = []

    algorithm = None
    for dim in [5]:
        budget = 2000 * dim
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:  # , 4, 5]
                problem = get_problem(fid, iid, dim)
                problem.attach_logger(l2)

                for rep in range(3):
                    np.random.seed(rep)
                    try:
                        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                        algorithm(problem)
                    except OverBudgetException:
                        pass

                    auc = correct_aoc(problem, l2, budget)
                    aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)

    return solution


class TestInitialPopulation(unittest.TestCase):
    def test_population_equal(self):
        # Instatntiate
        # gemini_key = os.getenv("GEMINI_API_KEY")
        llm = Dummy_LLM()

        # Instantiate and run object, to generate total population
        # of size 2.
        es1 = LLaMEA(
            evaluateBBOB,
            llm,
            n_parents=1,
            budget=1,
            experiment_name="Test-init-known-pop-same-size",
        )
        es1.run()
        dirname = es1.logger.dirname

        # Sleep for a second
        time.sleep(1)
        # Instantiate another run, with initial population.
        es2 = LLaMEA(
            evaluateBBOB,
            llm,
            n_parents=1,
            budget=1,
            experiment_name="Test-init-known-pop-same-size",
        )
        es2.run(dirname)
        dirname2 = es2.logger.dirname

        # Test consistency.
        pop1 = []
        pop2 = []
        with jsonlines.open(f"{dirname}/log.jsonl") as reader:
            for obj in reader:
                pop1.append(obj)

        with jsonlines.open(f"{dirname2}/log.jsonl") as reader:
            for obj in reader:
                pop2.append(obj)

        pop2 = pop2[: len(pop1)]

        # Check if all restored properties are equal.
        for index in range(len(pop1)):
            for property in [
                "code",
                "name",
                "description",
                "configspace",
                "operator",
                "task_prompt",
            ]:
                self.assertEqual(pop1[index][property], pop2[index][property])

    def test_population_greater(self):
        # Instatntiate
        llm = Dummy_LLM()

        # Instantiate and run object, to generate total population
        # of size 2.
        es1 = LLaMEA(
            evaluateBBOB,
            llm,
            n_parents=2,
            budget=5,
            experiment_name="Test-init-known-pop-same-size",
        )
        es1.run()
        dirname = es1.logger.dirname

        time.sleep(1)
        # Instantiate another run, with initial population.
        es2 = LLaMEA(
            evaluateBBOB,
            llm,
            n_parents=2,
            budget=5,
            experiment_name="Test-init-known-pop-same-size",
        )
        es2.run(dirname)
        dirname2 = es2.logger.dirname

        # Test consistency.
        pop1 = []
        pop2 = []
        with jsonlines.open(f"{dirname}/log.jsonl") as reader:
            for obj in reader:
                pop1.append(obj)

        with jsonlines.open(f"{dirname2}/log.jsonl") as reader:
            for obj in reader:
                pop2.append(obj)

        pop2 = pop2[: es2.n_parents]
        pop1 = pop1[-es1.n_parents :]
        # Check if all restored properties are equal.
        for index in range(len(pop1)):
            for property in [
                "code",
                "name",
                "description",
                "configspace",
                "operator",
                "task_prompt",
            ]:
                self.assertEqual(pop1[index][property], pop2[index][property])

    def test_lower_population(self):
        # Instatntiate
        llm = Dummy_LLM()

        # Instantiate and run object, to generate total population
        # of size 2.
        es1 = LLaMEA(
            evaluateBBOB,
            llm,
            n_parents=1,
            budget=1,
            experiment_name="Test-init-known-pop-same-size",
        )
        es1.run()
        dirname = es1.logger.dirname

        time.sleep(1)
        # Instantiate another run, with initial population.
        es2 = LLaMEA(
            evaluateBBOB,
            llm,
            n_parents=2,
            budget=1,
            experiment_name="Test-init-known-pop-same-size",
        )
        es2.run(dirname)
        dirname2 = es2.logger.dirname

        # Test consistency.
        pop1 = []
        pop2 = []
        with jsonlines.open(f"{dirname}/log.jsonl") as reader:
            for obj in reader:
                pop1.append(obj)

        with jsonlines.open(f"{dirname2}/log.jsonl") as reader:
            for obj in reader:
                pop2.append(obj)

        derivative_pop2 = pop2[: es1.n_parents]
        pop1 = pop1[-es1.n_parents :]
        # Check if all restored properties are equal.
        for index in range(len(pop1)):
            for property in [
                "code",
                "name",
                "description",
                "configspace",
                "operator",
                "task_prompt",
            ]:
                self.assertEqual(
                    pop1[index][property], derivative_pop2[index][property]
                )

        self.assertGreater(len(pop2), len(pop1))
