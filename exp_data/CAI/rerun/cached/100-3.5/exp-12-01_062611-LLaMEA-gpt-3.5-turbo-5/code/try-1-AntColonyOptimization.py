import numpy as np

class AntColonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evaluations = [func(ind) for ind in population]
        best_solution = population[np.argmin(evaluations)]
        
        return best_solution