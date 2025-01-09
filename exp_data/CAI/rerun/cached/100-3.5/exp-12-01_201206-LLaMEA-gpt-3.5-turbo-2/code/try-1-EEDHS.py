import numpy as np
from scipy.optimize import differential_evolution

class EEDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dimension = dim
        self.pop_size = 10
        self.sub_budget = int(budget / self.pop_size)

    def __call__(self, func):
        def ensemble_fitness(x):
            return np.mean([func(x) for _ in range(self.pop_size)])

        best_solution = None
        best_fitness = np.inf

        for _ in range(self.pop_size):
            initial_solution = np.random.uniform(-5.0, 5.0, self.dimension)
            result = differential_evolution(ensemble_fitness, bounds=[(-5.0, 5.0)]*self.dimension, maxiter=self.sub_budget)
            
            if result.fun < best_fitness:
                best_fitness = result.fun
                best_solution = result.x
        
        return best_solution