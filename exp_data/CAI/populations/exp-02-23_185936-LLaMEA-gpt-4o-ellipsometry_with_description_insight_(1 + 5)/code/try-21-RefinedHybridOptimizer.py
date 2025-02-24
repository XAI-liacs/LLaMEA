import numpy as np
from scipy.optimize import minimize

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        # Uniformly sample initial points within the defined bounds
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Dynamic adjustment of bounds for local refinement
            local_bounds = [(max(b[0], guess[i] - 0.1 * (b[1] - b[0])), min(b[1], guess[i] + 0.1 * (b[1] - b[0]))) for i, b in enumerate(bounds)]

            # Use the BFGS algorithm for local optimization with adapted bounds
            result = minimize(func, guess, method='L-BFGS-B', bounds=local_bounds, options={'maxfun': self.budget - evaluations})
            evaluations += result.nfev

            # Update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution