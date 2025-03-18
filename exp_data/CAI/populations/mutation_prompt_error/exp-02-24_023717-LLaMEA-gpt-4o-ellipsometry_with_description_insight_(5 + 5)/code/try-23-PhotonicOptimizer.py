import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Sobol sequence for initial guesses
        num_initial_guesses = min(10, self.budget // 10)  # Change 1: Increase initial guesses from 7 to 10
        sampler = Sobol(d=self.dim, scramble=False)
        initial_guesses = lb + (ub - lb) * sampler.random(num_initial_guesses)

        for initial_guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Local optimization using L-BFGS-B with constraints
            result = minimize(lambda x: self.evaluate(func, x, evaluations), 
                              initial_guess, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)], options={'maxiter': self.budget - evaluations})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution

    def evaluate(self, func, x, evaluations):
        if evaluations < self.budget:
            return func(x)
        else:
            raise Exception("Budget exceeded")

# Example usage:
# optimizer = PhotonicOptimizer(budget=100, dim=2)
# best_solution = optimizer(some_black_box_function)