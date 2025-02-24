import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class ABALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        initial_guesses = sobol_sampler.random_base2(m=int(np.log2(self.dim))) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
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
                best_solution - np.abs(guess - bounds[:, 0]) / 4,  # Adjust step size
                best_solution + np.abs(best_solution - bounds[:, 1]) / 2
            ]).T, func.bounds.lb, func.bounds.ub)
        
        return best_solution