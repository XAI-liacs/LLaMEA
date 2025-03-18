import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        # Using adaptive bounds for initial guesses
        initial_guesses = np.array([
            np.random.uniform(
                low=max(b[0], np.mean([func.bounds.lb[i], func.bounds.ub[i]]) - (b[1] - b[0]) / 4),
                high=min(b[1], np.mean([func.bounds.lb[i], func.bounds.ub[i]]) + (b[1] - b[0]) / 4),
                size=self.dim
            ) for b in [bounds] * num_initial_guesses
        ])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Use the BFGS algorithm for local optimization
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations})
            evaluations += result.nfev

            # Update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution