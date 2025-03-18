import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 8, 12)  # Increased samples to enhance initial exploration

        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0
        epsilon = 5e-7  # More stringent convergence threshold

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Introduced dynamic adjustment of `ftol` based on remaining budget
            dynamic_ftol = max(epsilon, (self.budget - evaluations) * 1e-7)
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations, 'ftol': dynamic_ftol})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                if best_value < epsilon:
                    break

        return best_solution