import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Phase 1: Uniform sampling for initial guesses
        samples = np.random.uniform(lb, ub, (self.budget // 2, self.dim))
        
        for sample in samples:
            if evaluations >= self.budget:
                break
            result = func(sample)
            evaluations += 1
            if result < best_value:
                best_value = result
                best_solution = sample

        # Phase 2: BFGS local optimization with fallback
        if best_solution is not None:
            def wrapped_func(x):
                nonlocal evaluations
                if evaluations >= self.budget:
                    return float('inf')
                evaluations += 1
                return func(x)

            bounds = [(max(lb[i], ub[i]), min(ub[i], lb[i])) for i in range(self.dim)]
            result = minimize(wrapped_func, best_solution, method='BFGS', bounds=bounds)
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            elif evaluations < self.budget:  # Fallback to Nelder-Mead
                result = minimize(wrapped_func, best_solution, method='Nelder-Mead')
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x

        return best_solution, best_value