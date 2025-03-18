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
            # Adaptive Restart with dynamic sampling
            num_samples = 5 if remaining_budget > self.budget // 2 else 3
            # Weighted random sampling for initial guesses
            weights = np.linspace(0.1, 1.0, num_samples)
            initial_guesses = [np.array([np.random.uniform(low, high) for low, high in bounds]) for _ in range(num_samples)]
            initial_guess = min(initial_guesses, key=lambda g: func(g) * np.random.choice(weights))

            # Dynamic local budget allocation with safety margin
            local_budget = max(5, remaining_budget // 3)

            # BFGS optimization with bounds and limited evaluations
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            elif remaining_budget < self.budget // 4:  # Adaptive restart mechanism
                remaining_budget = self.budget // 2

            remaining_budget -= result.nfev

        return best_solution, best_value