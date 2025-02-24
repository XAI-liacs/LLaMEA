import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guesses = self.informed_sampling(bounds, num_samples=min(10, self.budget // 4))
        best_solution = None
        best_value = float('inf')
        
        for init_guess in initial_guesses:
            result = minimize(
                func, init_guess, method='Nelder-Mead',
                options={'maxiter': self.budget // len(initial_guesses), 'adaptive': True, 'xatol': 1e-6}
            )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                bounds = self.iterative_bound_adjustment(bounds, best_solution + np.random.normal(0, 0.01, size=best_solution.shape))

        return best_solution
    
    def informed_sampling(self, bounds, num_samples):
        mean_point = np.mean(bounds, axis=0)
        cov_matrix = np.diag((bounds[1] - bounds[0]) / 10)
        return [np.clip(np.random.multivariate_normal(mean_point, cov_matrix), bounds[0], bounds[1]) for _ in range(num_samples)]
    
    def iterative_bound_adjustment(self, bounds, best_solution):
        lower_bound, upper_bound = bounds
        adjustment_ratio = 0.05
        new_bounds = np.vstack([
            np.maximum(lower_bound, best_solution - adjustment_ratio * (upper_bound - lower_bound)),
            np.minimum(upper_bound, best_solution + adjustment_ratio * (upper_bound - lower_bound))
        ])
        return new_bounds