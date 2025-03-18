import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        def adjust_bounds(solution):
            return [(max(b[0], s - 0.1 * (b[1] - b[0])), min(b[1], s + 0.1 * (b[1] - b[0]))) for s, b in zip(solution, bounds)]
        
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            adaptive_bounds = adjust_bounds(guess)
            result = minimize(func, guess, method='Powell', bounds=adaptive_bounds, options={'maxfev': self.budget - evaluations})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution