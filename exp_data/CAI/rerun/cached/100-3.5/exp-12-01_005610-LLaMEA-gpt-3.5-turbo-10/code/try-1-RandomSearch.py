import numpy as np

class RandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            fitness = func(solution)
            if fitness < best_fitness:
                best_solution = solution
                best_fitness = fitness
        return best_solution