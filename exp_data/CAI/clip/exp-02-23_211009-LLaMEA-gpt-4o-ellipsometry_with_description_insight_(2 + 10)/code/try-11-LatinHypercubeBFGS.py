import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class LatinHypercubeBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Extract bounds for the problem
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_diff = ub - lb

        # Number of initial points using a fraction of the budget
        num_initial_points = min(5, self.budget // 2)

        # Generate initial guesses using Latin Hypercube Sampling within bounds
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=num_initial_points)
        initial_guesses = lb + sample * range_diff

        best_solution = None
        best_value = float('inf')

        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break

            # Use BFGS for local optimization
            result = minimize(func, guess, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution