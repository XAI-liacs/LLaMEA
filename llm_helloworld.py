import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")

# Round 1
# messages = [
#     # {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "9.11 and 9.8, which is greater?"}
# ]
messages = [{"role": "user",
             "content": "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems.\nThe optimization algorithm should be able to find high-performing solutions to a wide range of tasks, which include evaluation on real-world applications such as, e.g., optimization of multilayered photonic structures. The ellipsometry problem involves retrieving the material and thickness of a reference layer by matching its reflectance properties using a known spectral response. The optimization minimizes the difference between the calculated and measured ellipsometric parameters for wavelengths between 400 and 800 nm and a fixed incidence angle of 40°. The parameters to be optimized include the thickness (30 to 250 nm) and refractive index (1.1 to 3) of the test layer. This relatively straightforward problem models a practical scenario where photonics researchers fine-tune a small number of parameters to achieve a desired spectral fit. This problem has small parameter space with fewer variables (thickness and refractive index), and the cost function is smooth and relatively free of noise, making it amenable to local optimization methods. Here are suggestions for designing algorithms: 1. Use local optimization algorithms like BFGS or Nelder-Mead, as they perform well in low-dimensional, smooth landscapes. 2. Uniform sampling across the parameter space ensures sufficient coverage for initial guesses. 3. Utilize fast convergence algorithms that can quickly exploit the smooth cost function landscape. 4. Iteratively adjust bounds and constraints to improve parameter estimates once initial solutions are obtained. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between func.bounds.lb (lower bound) and func.bounds.ub (upper bound). The dimensionality can be varied.\nGive an excellent and novel heuristic algorithm to solve this task and include it's one-line description with the main idea of the algorithm.\n\nProvide the Python code and a one-line description with the main idea (without enters). Give the response in the format:\n# Description: <short-description>\n# Code: \n```python\n<code>\n```\n"}]
try:
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    print(f"Reasoning: {reasoning_content}")
    print(f"Response: {content}")
except Exception as e:
    if hasattr(e, 'response'):
        print(f"HTTP 状态码: {e.response.status_code}")
        print(f"原始响应内容: {e.response.text}")
    else:
        print(f"错误: {e}")
