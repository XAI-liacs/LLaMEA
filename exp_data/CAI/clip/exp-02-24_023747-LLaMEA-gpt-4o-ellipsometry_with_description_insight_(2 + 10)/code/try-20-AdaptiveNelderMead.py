import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guesses = self.mean_centered_sampling(bounds, num_samples=min(10, self.budget // 4))
        best_solution = None
        best_value = float('inf')
        
        for init_guess in initial_guesses:
            result = minimize(
                func, init_guess, method='Nelder-Mead',
                options={'maxiter': self.budget // len(initial_guesses), 'fatol': 1e-8}  # changed tolerance for faster convergence
            )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Adjust bounds based on new best solution
                bounds = self.iterative_bound_adjustment(bounds, best_solution)

        return best_solution
    
    def mean_centered_sampling(self, bounds, num_samples):
        mean_point = np.mean(bounds, axis=0)
        return [mean_point + np.random.uniform(-0.5, 0.5, size=mean_point.shape) * (bounds[1] - bounds[0]) for _ in range(num_samples)]
    
    def iterative_bound_adjustment(self, bounds, best_solution):
        lower_bound, upper_bound = bounds
        new_bounds = np.vstack([
            np.maximum(lower_bound, best_solution - 0.1 * (upper_bound - lower_bound)),
            np.minimum(upper_bound, best_solution + 0.1 * (upper_bound - lower_bound))
        ])
        return new_bounds