import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class NaturalComputingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget

        # Initial Sobol sampling for better exploration
        num_initial_samples = min(self.dim * 5, remaining_budget // 2)
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        samples = sobol_sampler.random_base2(m=int(np.log2(num_initial_samples)))
        samples = lb + samples * (ub - lb)  # Scale samples to [lb, ub]
        
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if remaining_budget <= 0:
                break

        # Local optimization using L-BFGS-B with dynamic tolerance
        if remaining_budget > 0:
            options = {'maxfun': remaining_budget, 'ftol': max(1e-9, best_value * 1e-6)}
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=options)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution