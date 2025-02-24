import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedABALS:
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
        cov_matrix = np.eye(self.dim) * 0.1  # Initial covariance matrix for adaptation
        
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            
            # Local optimization using L-BFGS-B
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update the covariance matrix for boundary adaptation
            diff = result.x - guess
            cov_matrix = 0.9 * cov_matrix + 0.1 * np.outer(diff, diff)

            # Dynamically adjust the bounds based on a weighted combination of the best and worst solutions
            adaptive_offset = np.sqrt(np.diagonal(cov_matrix))
            bounds = np.clip(np.array([
                result.x - adaptive_offset * 0.5,  # Modified line to adjust bounds more conservatively
                result.x + adaptive_offset * 0.5   # Modified line to adjust bounds more conservatively
            ]).T, func.bounds.lb, func.bounds.ub)
        
        return best_solution