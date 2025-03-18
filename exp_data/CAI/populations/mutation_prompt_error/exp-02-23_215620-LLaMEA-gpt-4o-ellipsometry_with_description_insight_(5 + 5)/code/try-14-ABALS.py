import numpy as np
from scipy.optimize import minimize

class ABALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guesses = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim * 2, self.dim))  # Change 1

        best_solution = None
        best_value = np.inf
        
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Dynamically adjust the bounds based on the current best solution
            bounds = np.clip(np.array([
                (0.7 * best_solution + 0.3 * bounds[:, 0]),  # Change 2
                (0.7 * best_solution + 0.3 * bounds[:, 1])   # Change 3
            ]).T, func.bounds.lb, func.bounds.ub)
        
        return best_solution