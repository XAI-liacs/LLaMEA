import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Uniform sampling for initial guesses
        num_initial_samples = min(5, self.budget // 5)
        best_solution = None
        best_value = float('inf')

        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub

        # Function wrapper to count evaluations
        def wrapped_func(x):
            if self.evals >= self.budget:
                raise ValueError("Budget exceeded")
            self.evals += 1
            return func(x)

        # Generate initial valid points within the bounds
        initial_solutions = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        for i in range(num_initial_samples):
            result = minimize(wrapped_func, initial_solutions[i], method='Nelder-Mead', bounds=np.array([lb, ub]).T)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Adaptively adjust bounds if we are not close to the budget
            if self.evals < self.budget * 0.75:
                lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))

        return best_solution