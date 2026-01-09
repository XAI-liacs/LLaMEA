from unittest.mock import MagicMock

import pytest
import json

from llamea import LLaMEA, Ollama_LLM, Solution, LLM

# Helper
class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)


def test_algorithm_generation():
    """Test the algorithm generation process."""

    def f(solution):
        return f"feedback {solution.name}", 1.0, "", {}

    optimizer = LLaMEA(
        f, llm=Ollama_LLM("model"), experiment_name="test generation", log=False
    )
    response = "# Description: Long Example Algorithm\n# Code:\n```python\nclass ExampleAlgorithm:\n    pass\n```"
    optimizer.llm.query = MagicMock(return_value=response)

    individual = optimizer.llm.sample_solution(
        session_messages=[{"role": "system", "content": "test prompt"}]
    )

    assert (
        individual.description == "Long Example Algorithm"
    ), f"Algorithm long name should be extracted correctly, is {individual.description}"
    assert (
        individual.name == "ExampleAlgorithm"
    ), "Algorithm name should be extracted correctly"
    assert (
        "class ExampleAlgorithm" in individual.code
    ), "Algorithm code should be extracted correctly"


def test_evolve_solution_with_diff():
    def f(sol, logger):
        return sol

    optimizer = LLaMEA(
        f,
        llm=Ollama_LLM("model"),
        experiment_name="diff",
        log=False,
        diff_mode=True,
        evaluate_population=False,
    )

    base = Solution(code="class MyAlgo:\n    pass\n", name="MyAlgo", description="d")
    optimizer.population = [base]
    diff_reply = """
```
<<<<<<< SEARCH
    pass
=======
    # Added loops
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE
```"""
    optimizer.llm.query = MagicMock(return_value=diff_reply)
    evolved = optimizer.evolve_solution(base)
    assert "# Added loops" in evolved.code, evolved.code + " does not contain added code '# Added loops'"
    assert "pass" not in evolved.code, evolved.code + " contains code 'pass'"
    assert "<<<<<<< SEARCH" not in evolved.code, evolved.code + " contains Search marker."
    assert ">>>>>>> REPLACE" not in evolved.code, evolved.code + " contains Replace marker."

class MockLLM(LLM):
    
    def __init__(self, model):
        super().__init__(api_key='unknown', model=model)
        self.request_messages = []
        self.response_messages = []
        with open('exp-failed/conversationlog.jsonl') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if data['role'] == 'client':
                    self.request_messages.append(data['content'])
                else:
                    self.response_messages.append(data['content'])
    
    def query(self, session):
        reply = self.response_messages.pop(0)
        return reply
    
    def get_expected_query(self):
        request = self.request_messages.pop(0)
        return [{'role': 'client', 'content': request}]


def evaluator_generator():
    evolution_data = []
    evolution_index = 0
    with open('exp-failed/log.jsonl') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            evolution_data.append(data)
    
    def evaluate(solution, logger=None):
        nonlocal evolution_index
        fitness = evolution_data[evolution_index]['fitness']
        solution.fitness = float(fitness)
        evolution_index += 1
        print(f'Evolution Index: {evolution_index}')
        if "<<<<<<< SEARCH" in solution.code:
            raise ValueError(solution.code + " contains Search marker.")
        if ">>>>>>> REPLACE" in solution.code:
            raise ValueError(solution.code + " contains Replace marker.")
        return solution
    return evaluate, len(evolution_data)



def test_complete_evaluation_diff_mode(monkeypatch):
    from llamea.utils import apply_code_delta
    import re
    evaluator, budget = evaluator_generator()
    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    def patched_sample_solution(
        self,
        session_messages: list,
        parent_ids: list | None = None,
        HPO: bool = False,
        base_code: str | None = None,
        diff_mode: bool = False,
    ):
        session_messages = llm.get_expected_query()
        if parent_ids is None:
            parent_ids = []

        if self.log:
            self.logger.log_conversation(
                "client", "\n".join([d["content"] for d in session_messages])
            )

        message = self.query(session_messages)
        if self.log:
            self.logger.log_conversation(self.model, message)

        code_block = self.extract_algorithm_code(message)
        code = ""
        success = False  # <- Flag to Implement fall back to code block update, when LLM fails to adhere to diff mode.
        if diff_mode:
            if base_code is None:
                base_code = ""
            else:
                code, success, similarity = apply_code_delta(code_block, base_code)
                print(
                    f"\t Diff application {'un' if not success else ''}successful, Similarity {similarity * 100:.2f}%."
                )
        else:
            code = code_block

        if diff_mode and not success:
            print("\t\t Falling back to code replace.")
            code = code_block

        name = re.findall(
            r"(?:def|class)\s*(\w*).*\:",
            code,
            re.IGNORECASE,
        )[0]
        desc = self.extract_algorithm_description(message)
        cs = None
        if HPO and ConfigurationSpace is not None:
            cs = self.extract_configspace(message)
        new_individual = Solution(
            name=name,
            description=desc,
            configspace=cs,
            code=code,
            parent_ids=parent_ids,
        )
        return new_individual
    monkeypatch.setattr(LLM, "sample_solution", patched_sample_solution)
    llm = MockLLM(model='Bruh-LLM')
    
    llamea = LLaMEA(evaluator,
                      llm, 
                      budget=budget,
                      mutation_prompts=mutation_prompts, 
                      n_parents=8, 
                      n_offspring=8,
                      max_workers=1,
                      elitism=True, 
                      adaptive_prompt=True, 
                      diff_mode=True)
    
    llamea.run()
    