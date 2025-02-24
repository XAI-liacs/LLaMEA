import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class DynamicNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = float('inf')

        # Latin Hypercube Sampling for initial guesses
        num_initial_guesses = min(5, self.budget // self.dim)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=num_initial_guesses)
        initial_guesses = qmc.scale(sample, bounds[0], bounds[1])

        for guess in initial_guesses:
            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='Nelder-Mead',
                              options={'maxfev': self.budget - self.evaluations, 'adaptive': True})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Break early if budget exceeded
            if self.evaluations >= self.budget:
                break

            # Dynamically adjust simplex size based on best solution found
            simplex_size = 0.05 * (bounds[1] - bounds[0])
            bounds[0] = np.maximum(func.bounds.lb, best_solution - simplex_size)
            bounds[1] = np.minimum(func.bounds.ub, best_solution + simplex_size)

        return best_solution

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            # Avoid further evaluations and terminate early
            return float('inf')