import numpy as np
from scipy.optimize import minimize

class TrustRegionOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        search_space = np.array([ub[i] - lb[i] for i in range(self.dim)])
        remaining_budget = self.budget

        # Random sampling for initial candidate solutions
        num_samples = min(5, self.budget // 10)
        initial_guesses = np.random.uniform(lb, ub, size=(num_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')

        # Evaluate initial guesses
        evaluations = 0
        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            value = func(guess)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = guess

        # Use Trust-Region for local optimization starting from best initial guess
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            value = func(x)
            evaluations += 1
            return value
        
        result = minimize(wrapped_func, best_solution, method='trust-constr', bounds=list(zip(lb, ub)))
        
        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        # Adaptive bound contraction for final refinement
        if evaluations < self.budget:
            contracted_bounds = [(max(lb[i], best_solution[i] - 0.1 * search_space[i]), 
                                  min(ub[i], best_solution[i] + 0.1 * search_space[i])) for i in range(self.dim)]
            result = minimize(wrapped_func, best_solution, method='trust-constr', bounds=contracted_bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution