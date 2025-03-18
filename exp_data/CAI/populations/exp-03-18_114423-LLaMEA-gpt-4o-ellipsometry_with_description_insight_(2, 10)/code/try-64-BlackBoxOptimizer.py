import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_solution = None
        best_value = np.inf
        remaining_budget = self.budget

        while remaining_budget > 0:
            num_samples = max(3, remaining_budget // self.budget * 10)
            initial_guesses = [np.array([np.random.uniform(low, high) for low, high in bounds]) for _ in range(num_samples)]
            initial_guess = min(initial_guesses, key=lambda g: func(g))

            # Adjust local budget allocation to be more adaptive
            local_budget = max(5, int(remaining_budget * 0.3))  # Adjusted allocation based on remaining budget

            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            # Restart mechanism added to enhance solution diversity
            elif np.random.rand() < 0.1:
                initial_guess = np.array([np.random.uniform(low, high) for low, high in bounds])

            remaining_budget -= result.nfev

        return best_solution, best_value