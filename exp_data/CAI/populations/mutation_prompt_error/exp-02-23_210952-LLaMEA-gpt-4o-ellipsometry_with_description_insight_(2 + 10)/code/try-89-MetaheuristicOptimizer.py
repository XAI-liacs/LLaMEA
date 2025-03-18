import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        num_initial_samples = max(self.budget // 3, 5)
        remaining_budget = self.budget - num_initial_samples

        sobol_engine = Sobol(d=self.dim, scramble=True)
        initial_solutions = sobol_engine.random_base2(m=int(np.log2(num_initial_samples)))
        initial_solutions = lower_bounds + (upper_bounds - lower_bounds) * initial_solutions

        best_solution = None
        best_score = float('inf')

        for solution in initial_solutions:
            score = func(solution)
            if score < best_score:
                best_score = score
                best_solution = solution

        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        improvement_threshold = 1e-6  # Stopping criterion based on improvement
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget, 'ftol': improvement_threshold})

        return result.x if result.success else best_solution