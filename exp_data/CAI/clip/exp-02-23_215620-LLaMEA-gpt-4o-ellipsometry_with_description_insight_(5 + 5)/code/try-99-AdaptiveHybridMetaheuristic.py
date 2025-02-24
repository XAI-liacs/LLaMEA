import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        initial_guesses = sobol_sampler.random_base2(m=int(np.log2(self.budget))) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        best_solution = None
        best_value = np.inf
        cov_matrix = np.eye(self.dim) * 0.1  # Initial covariance matrix for adaptation
        
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break

            # Local optimization using L-BFGS-B with adaptive tolerance
            tol = 1e-6 + 1e-2 * (self.evaluations / self.budget)
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations, 'ftol': tol})
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update covariance matrix and dynamically adjust bounds
            diff = result.x - guess
            cov_matrix = 0.9 * cov_matrix + 0.1 * np.outer(diff, diff)
            adaptive_offset = np.sqrt(np.diagonal(cov_matrix))
            bounds = np.clip(np.array([
                best_solution - adaptive_offset,
                best_solution + adaptive_offset
            ]).T, func.bounds.lb, func.bounds.ub)
        
        return best_solution