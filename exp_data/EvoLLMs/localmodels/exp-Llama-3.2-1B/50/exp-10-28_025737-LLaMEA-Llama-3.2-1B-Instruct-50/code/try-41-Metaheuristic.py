import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Refine the strategy
        if len(self.search_space) < 10:
            self.search_space = [x for x in self.search_space if x not in best_func]
        elif random.random() < 0.45:
            self.search_space = [x for x in self.search_space if x not in best_func]

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

# Test the algorithm
budget = 100
dim = 10
algorithm = NovelMetaheuristicAlgorithm(budget, dim)
func = lambda x: x**2
best_func = algorithm(func)
print("Best function:", best_func)