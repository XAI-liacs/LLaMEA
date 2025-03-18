import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guesses = self.stratified_sampling(bounds, num_samples=min(10, self.budget // 4))
        best_solution = None
        best_value = float('inf')
        
        for init_guess in initial_guesses:
            result = minimize(
                func, init_guess, method='Nelder-Mead',
                options={'maxiter': self.budget // len(initial_guesses), 'fatol': 1e-5}
            )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                bounds = self.iterative_bound_adjustment(bounds, best_solution)

        return best_solution
    
    def stratified_sampling(self, bounds, num_samples):
        return [np.random.uniform(bounds[0], bounds[1]) for _ in range(num_samples)]
    
    def iterative_bound_adjustment(self, bounds, best_solution):
        lower_bound, upper_bound = bounds
        range_adjustment = 0.05 * (upper_bound - lower_bound)
        new_bounds = np.vstack([
            np.maximum(lower_bound, best_solution - range_adjustment),
            np.minimum(upper_bound, best_solution + range_adjustment)
        ])
        return new_bounds