import os
import re
import torch
import openai
import difflib
import transformers
import numpy as np
from llamea import ExperimentLogger, Individual

if torch.cuda.is_available():
    print(
        f"CUDA is available. PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device index: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Using CPU.")

role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
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
output_format_prompt = """
Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def code_compare(code1, code2):
    modified_lines = 0
    total_lines = max(len(code1), len(code2))
    matcher = difflib.SequenceMatcher(None, code1, code2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            modified_lines += max(i2 - i1, j2 - j1)
        elif tag in ('delete', 'insert'):
            modified_lines += (i2 - i1) if tag == 'delete' else (j2 - j1)
    diff_ratio = modified_lines / total_lines if total_lines > 0 else 0
    return diff_ratio


def extract_algorithm_code(message):
    pattern = r"```(?:python)?\n(.*?)\n```"
    match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        raise NoCodeException


def extract_algorithm_description(message):
    pattern = r"#\s*Description\s*:\s*(.*)"
    match = re.search(pattern, message, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return ""


def construct_prompt(individual, mutation_prompt):
    solution = individual.solution
    description = individual.description
    feedback = individual.feedback
    num_lines = len(solution.split("\n"))
    prob = 0.2
    mutation_operator = f"""
Now, refine the strategy of the selected solution to improve it. Make sure you 
only change {(prob*100):.1f}% of the code, which means if the code has 100 lines, you 
can only change {prob*100} lines, and the rest lines should remain the same. For 
this code, it has {num_lines} lines, so you can only change {max(1, int(prob*num_lines))}
lines, the rest {num_lines-max(1, int(prob*num_lines))} lines should remain the same. 
This changing rate {(prob*100):.1f}% is the mandatory requirement, you cannot change 
more or less than this rate.
"""
    individual.set_mutation_prompt(mutation_operator)
    final_prompt = f"""{task_prompt}
The selected solution to update is:
{description}

With code:
{solution}

{feedback}

{mutation_operator}
{output_format_prompt}
"""
    session_messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": final_prompt},
    ]
    return session_messages


def llm_chat(client, model_name, session_messages, logger, generation,
             parent_id=None):
    logger.log_conversation(
        "LLaMEA", "\n".join([d["content"] for d in session_messages]))
    if "gpt" in model_name:
        response = client.chat.completions.create(
            model=model_name, messages=session_messages, temperature=0.8)
        message = response.choices[0].message.content
    elif "Llama" in model_name:
        history = []
        for msg in session_messages:
            if msg["role"] == "LLaMEA":
                role = "user"
            elif "Llama" in msg["role"]:
                role = "assistant"
            else:
                role = msg["role"]
            history.append(
                {
                    "role": role,
                    "content": msg["content"]
                }
            )
        response = client(history, max_new_tokens=8192)
        message = response[0]["generated_text"][-1]['content']
    logger.log_conversation(model_name, message)
    code = extract_algorithm_code(message)
    name = re.findall(
        "class\\s*(\\w*)(?:\\(\\w*\\))?\\:",
        code,
        re.IGNORECASE,
    )[0]
    desc = extract_algorithm_description(message)
    individual = Individual(code, name, desc, None, generation, parent_id)
    logger.log_code(generation, name, code)
    return individual


def mutation_on_same_code(model_name="gpt-4o", experiment_name=None,
                          mutation_prompt=None, budget=100):
    logger = ExperimentLogger(f"LLaMEA-{model_name}-{experiment_name}")
    session_messages = [
        {"role": "system", "content": role_prompt},
        {
            "role": "user",
            "content": task_prompt + output_format_prompt,
        },
    ]
    if "gpt" in model_name:
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
    elif "Llama" in model_name:
        model_id = f"meta-llama/{model_name}"
        client = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    init_individual = llm_chat(client, model_name, session_messages, logger, 0)
    generation = 1
    code_diffs = []
    for _ in range(budget):
        print(f"{logger.dirname}: generation {generation}")
        new_prompt = construct_prompt(init_individual, mutation_prompt)
        evolved_individual = init_individual.copy()
        try:
            evolved_individual = llm_chat(client, model_name, new_prompt,
                                          logger, generation,
                                          init_individual.id)
        except NoCodeException:
            print(
                "No code was extracted. The code should be encapsulated with ``` in your response.",
                "The code should be encapsulated with ``` in your response.",
            )
        except Exception as e:
            error = repr(e)
            print(f"An exception occurred: {error}.", error)
        if evolved_individual is not None:
            code_diffs += [code_compare(init_individual.solution,
                                        evolved_individual.solution)]
        generation += 1
    return code_diffs


def mutation_on_different_code(model_name="gpt-4o", experiment_name=None,
                               mutation_prompt=None, budget=100):
    code_diffs = []
    for i in range(budget):
        logger = ExperimentLogger(f"LLaMEA-{model_name}-{experiment_name}-{i}")
        session_messages = [
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": task_prompt + output_format_prompt,
            },
        ]
        if "gpt" in model_name:
            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)
        elif "Llama" in model_name:
            model_id = f"meta-llama/{model_name}"
            client = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        init_individual = llm_chat(client, model_name, session_messages, logger, 0)
        generation = 1
        print(f"{logger.dirname}: generation {generation}")
        new_prompt = construct_prompt(init_individual, mutation_prompt)
        evolved_individual = init_individual.copy()
        try:
            evolved_individual = llm_chat(client, model_name, new_prompt,
                                          logger, generation,
                                          init_individual.id)
        except NoCodeException:
            print(
                "No code was extracted. The code should be encapsulated with ``` in your response.",
                "The code should be encapsulated with ``` in your response.",
            )
        except Exception as e:
            error = repr(e)
            print(f"An exception occurred: {error}.", error)
        if evolved_individual is not None:
            code_diffs += [code_compare(init_individual.solution,
                                        evolved_individual.solution)]
        generation += 1
    return code_diffs


def evalutate_mutation_prompt(model_name="gpt-4o", experiment_name=None,
                              mutation_prompt=None, budget=20):
    return mutation_on_same_code(model_name, experiment_name,
                                 mutation_prompt, budget)


def ape_lite():
    pass


# mutation_on_same_code("Llama-3.3-70B-Instruct", "20", None, 100)
mutation_on_different_code("gpt-3.5-turbo", "mutation_on_different_code_20", None, 100)
