import numpy as np

class AdaptiveHybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def de(func, bounds, pop_size=10, f=0.5, cr=0.9, max_iter=100):
            # DE implementation
            pass

        def pso(func, bounds, swarm_size=10, omega=0.5, phi_p=0.5, phi_g=0.5, max_iter=100):
            # PSO implementation
            pass

        current_budget = 0
        while current_budget < self.budget:
            de_budget = min(self.budget - current_budget, int(0.4 * self.budget))  # Allocate 40% budget to DE
            pso_budget = self.budget - current_budget - de_budget  # Allocate remaining budget to PSO

            de_best = de(func, bounds=[-5.0, 5.0], pop_size=10, max_iter=100)
            pso_best = pso(func, bounds=[-5.0, 5.0], swarm_size=10, max_iter=100)

            current_budget += de_budget + pso_budget

        return de_best if func(de_best) < func(pso_best) else pso_best