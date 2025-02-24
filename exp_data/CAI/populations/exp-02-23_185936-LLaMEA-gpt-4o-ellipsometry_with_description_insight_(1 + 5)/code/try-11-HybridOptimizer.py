import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        # Adaptive sampling: focus more initial guesses near the middle of the search space
        initial_guesses = np.array([np.random.uniform(low=(b[0] + b[1]) / 3, high=(2 * b[1] + b[0]) / 3, size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            # Use the BFGS algorithm for local optimization with modified tolerances
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-9, 'maxfun': self.budget - evaluations})
            evaluations += result.nfev

            # Update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution