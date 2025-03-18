import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guesses = self.uniform_sampling(bounds, num_samples=min(10, self.budget // 4))
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
                
        return best_solution
    
    def uniform_sampling(self, bounds, num_samples):
        return [np.random.uniform(low=bounds[0], high=bounds[1]) for _ in range(num_samples)]