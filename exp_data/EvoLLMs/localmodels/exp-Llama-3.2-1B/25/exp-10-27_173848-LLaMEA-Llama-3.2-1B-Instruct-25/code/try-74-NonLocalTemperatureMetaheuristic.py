# Description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the temperature
            self.temp = max(0.1, self.temp * 0.95)

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# ```python
import numpy as np

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                # Update the best function
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the temperature
            self.temp = max(0.1, self.temp * 0.95)

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# ```python
import numpy as np

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                # Update the best function
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the temperature
            self.temp = max(0.1, self.temp * 0.95)

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

# Example usage:
if __name__ == "__main__":
    # Define the problem
    func = lambda x: x**2
    budget = 10
    dim = 5

    # Initialize the metaheuristic
    metaheuristic = NonLocalTemperatureMetaheuristic(budget, dim)

    # Optimize the function
    best_func = metaheuristic(func)
    print(f"Optimized function: {best_func}")
    print(f"Best fitness: {best_func(x=0)}")