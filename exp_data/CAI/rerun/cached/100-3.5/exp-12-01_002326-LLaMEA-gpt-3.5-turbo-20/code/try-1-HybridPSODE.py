import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iters = budget // self.pop_size

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        for _ in range(self.max_iters):
            # PSO updates
            # DE updates
        best_solution = pop[np.argmin(func(pop))]
        return best_solution