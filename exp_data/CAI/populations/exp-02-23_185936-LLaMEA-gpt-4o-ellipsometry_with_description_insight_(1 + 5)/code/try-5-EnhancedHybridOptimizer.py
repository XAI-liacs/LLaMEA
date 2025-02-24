import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        # Use Latin Hypercube Sampling for an enhanced initial guess distribution
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_points = sampler.random(n=num_initial_guesses)
        initial_guesses = qmc.scale(initial_points, [b[0] for b in bounds], [b[1] for b in bounds])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Use the BFGS algorithm for local optimization
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations})
            evaluations += result.nfev

            # Update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Tighten bounds around the current best solution
                bounds = [(max(bounds[i][0], best_solution[i] - 0.1 * (bounds[i][1] - bounds[i][0])), 
                           min(bounds[i][1], best_solution[i] + 0.1 * (bounds[i][1] - bounds[i][0]))) 
                          for i in range(self.dim)]

        return best_solution