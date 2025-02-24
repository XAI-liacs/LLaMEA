import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from skopt import gp_minimize

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        # Initial Latin Hypercube sampling for better coverage
        sampler = qmc.LatinHypercube(d=self.dim)
        sample_points = sampler.random(n=16)
        initial_guesses = qmc.scale(sample_points, func.bounds.lb, func.bounds.ub).tolist()
        best_solution = None
        best_value = float('inf')

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Bayesian optimization for adaptive method selection
        def method_selector(x):
            if self.evaluation_count < self.budget / 2:
                return 'L-BFGS-B'
            else:
                return 'Nelder-Mead'

        # Begin with a local optimizer
        for guess in initial_guesses:
            method = method_selector(guess)
            result = minimize(wrapped_func, guess, method=method, bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

        return best_solution