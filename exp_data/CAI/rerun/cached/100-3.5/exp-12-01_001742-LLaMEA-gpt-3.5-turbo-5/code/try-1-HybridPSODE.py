import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.pso = ParticleSwarmOptimization(budget, dim)  
        self.de = DifferentialEvolution(budget, dim)

    def __call__(self, func):
        best_solution = np.random.uniform(low=-5.0, high=5.0, size=self.dim)
        for _ in range(self.budget):
            if np.random.rand() < 0.5:
                candidate = self.pso(func)
            else:
                candidate = self.de(func)
                
            if func(candidate) < func(best_solution):
                best_solution = candidate
                
        return best_solution