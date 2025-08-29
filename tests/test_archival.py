from logging import Logger
import unittest
import random
import time

from llamea import LLaMEA, Dummy_LLM, ExperimentLogger, Solution


def evaluationFunction(solution, explogger=None):
    score = random.random()
    solution.set_score(score, f"Score was {score}.")
    return solution


class TestArchival(unittest.TestCase):
    def test_archival(self):
        llm = Dummy_LLM()
        roleprompt = "You are a gamer"
        task_prompt = "Test prompt. You are asked to pick a number between 1-10, and will get response as to distance between correct and your answer. The correct answer changes every time."
        # Instantiate an object.

        es = LLaMEA(
            evaluationFunction,
            llm,
            n_parents=1,
            task_prompt=task_prompt,
            role_prompt=roleprompt,
            budget=200,
        )
        dirname = es.logger.dirname
        time.sleep(4)  # Wait for archival to reflect on storage.
        es_start_archive = LLaMEA.warm_start(dirname)  # Warm start.

        archived_es = es_start_archive.__dict__
        for key, value in es.__dict__.items():
            if isinstance(
                value, Logger | ExperimentLogger | Solution | None | Dummy_LLM
            ):  # Objects when resurrected will not have same identifier.
                pass
            else:
                self.assertEqual(archived_es[key], value)
