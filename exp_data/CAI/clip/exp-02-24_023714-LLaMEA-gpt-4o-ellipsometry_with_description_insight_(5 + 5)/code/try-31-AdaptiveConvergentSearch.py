import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        # Initial Sobol sequence sampling for better coverage
        sampler = Sobol(d=self.dim, scramble=True)
        initial_guesses = sampler.random_base2(m=5) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        best_solution = None
        best_value = float('inf')

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Begin with a local optimizer
        for guess in initial_guesses:
            result = minimize(wrapped_func, guess, method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.05 * (np.array(func.bounds.ub) - np.array(func.bounds.lb))) # Changed from 0.1 to 0.05
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.05 * (np.array(func.bounds.ub) - np.array(func.bounds.lb))) # Changed from 0.1 to 0.05

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

        return best_solution