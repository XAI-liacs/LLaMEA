import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        # Adjust the number of initial guesses based on budget 
        num_initial_guesses = max(1, min(self.budget // 5, 15))
        
        # Enhanced uniform sampling strategy for better coverage
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Adaptive L-BFGS-B with updated epsilon for improved local search
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations, 'eps': 1e-8})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution