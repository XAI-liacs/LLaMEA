import numpy as np
from scipy.optimize import minimize

class GradientInformedAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = float('inf')

        # Begin with uniform sampling for initial guesses
        num_initial_guesses = min(5, self.budget // self.dim)
        initial_guesses = np.random.uniform(bounds[0] + 0.1 * (bounds[1] - bounds[0]), 
                                            bounds[1] - 0.1 * (bounds[1] - bounds[0]), 
                                            (num_initial_guesses, self.dim))

        for guess in initial_guesses:
            # Use BFGS without constraints for the first pass to gather gradient information
            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='BFGS', options={'maxiter': self.budget // 2})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Break early if budget exceeded
            if self.evaluations >= self.budget:
                break

            # Use the gradient to dynamically adapt search bounds
            if result.jac is not None:
                gradient = result.jac
                adaptive_bounds_range = 0.2 * np.abs(gradient)
                bounds[0] = np.maximum(func.bounds.lb, best_solution - adaptive_bounds_range)
                bounds[1] = np.minimum(func.bounds.ub, best_solution + adaptive_bounds_range)

            # Refine search using L-BFGS-B within adapted bounds
            result = minimize(self.evaluate_func, best_solution, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': self.budget - self.evaluations})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            # Avoid further evaluations and terminate early
            return float('inf')