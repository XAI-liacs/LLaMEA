import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Uniform sampling for initial guesses
        num_initial_guesses = min(7, self.budget // 10)  # Change 1: Increase initial guesses from 5 to 7
        initial_guesses = np.random.uniform(lb, ub, (num_initial_guesses, self.dim))

        for initial_guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Local optimization using L-BFGS-B with constraints
            result = minimize(lambda x: self.evaluate(func, x, evaluations), 
                              initial_guess, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)], options={'maxiter': self.budget - evaluations, 'ftol': 1e-9})  # Change 2: Added convergence criteria 'ftol' for stability
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