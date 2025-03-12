import numpy as np
from scipy.optimize import minimize, approx_fprime

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guesses = self.adaptive_sampling(func, bounds, num_samples=min(10, self.budget // 4))
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
                # Adjust bounds based on new best solution
                bounds = self.iterative_bound_adjustment(bounds, best_solution)

        return best_solution
    
    def adaptive_sampling(self, func, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            random_point = np.random.uniform(bounds[0], bounds[1])
            gradient = approx_fprime(random_point, func, 1e-6)
            samples.append(random_point + 0.05 * gradient)
        return samples
    
    def iterative_bound_adjustment(self, bounds, best_solution):
        lower_bound, upper_bound = bounds
        new_bounds = np.vstack([
            np.maximum(lower_bound, best_solution - 0.1 * (upper_bound - lower_bound)),
            np.minimum(upper_bound, best_solution + 0.1 * (upper_bound - lower_bound))
        ])
        return new_bounds