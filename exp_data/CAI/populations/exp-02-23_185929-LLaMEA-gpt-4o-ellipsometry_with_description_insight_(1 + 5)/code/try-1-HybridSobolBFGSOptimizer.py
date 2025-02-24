import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridSobolBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget

        # Initial Sobol sequence sampling
        num_initial_samples = min(self.dim * 5, remaining_budget // 2)
        sobol_engine = Sobol(d=self.dim, scramble=True)
        samples = sobol_engine.random_base2(m=int(np.log2(num_initial_samples)))
        samples = lb + (ub - lb) * samples
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if remaining_budget <= 0:
                break

        # Local optimization using BFGS
        if remaining_budget > 0:
            result = minimize(func, best_solution, method='BFGS', options={'maxiter': remaining_budget})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution