import numpy as np
from scipy.optimize import minimize

class ABALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guesses = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim, self.dim))
        
        best_solution = None
        best_value = np.inf
        avg_solution = np.zeros(self.dim)
        count = 0
        
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            avg_solution += result.x
            count += 1

            # Dynamically adjust the bounds based on the current best and average solutions
            avg_solution /= count
            bounds = np.clip(np.array([
                best_solution - np.abs(avg_solution - bounds[:, 0]) / 2,
                best_solution + np.abs(avg_solution - bounds[:, 1]) / 2
            ]).T, func.bounds.lb, func.bounds.ub)
        
        return best_solution