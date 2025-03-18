import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class ABALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Use Sobol sequence for initial guesses
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guesses = sobol_sampler.random_base2(m=int(np.log2(self.dim))) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
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

            # Dynamically adjust the bounds based on the current best and average solutions with step size adjustment
            avg_solution /= count
            step_size = 0.5 * np.abs(avg_solution - bounds.mean(axis=1))
            bounds = np.clip(np.array([
                best_solution - step_size,
                best_solution + step_size
            ]).T, func.bounds.lb, func.bounds.ub)
        
        return best_solution