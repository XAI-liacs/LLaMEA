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
            num_samples = max(3, int(remaining_budget / self.budget * 20))  # Adjusted sample size
            initial_guesses = [np.array([np.random.uniform(low, high) for low, high in bounds]) for _ in range(num_samples)]
            initial_guess = min(initial_guesses, key=lambda g: func(g))

            local_budget = max(5, int(remaining_budget * 0.20))  # Adjusted budget allocation

            # Dynamic boundary adjustment based on current best_solution
            if best_solution is not None:
                adjusted_bounds = [(max(low, best_solution[i] - 0.1*(high-low)), 
                                    min(high, best_solution[i] + 0.1*(high-low))) for i, (low, high) in enumerate(bounds)]
            else:
                adjusted_bounds = bounds

            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=adjusted_bounds, options={'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            # Enhanced restart mechanism with probabilistic threshold
            elif np.random.rand() < 0.2:
                initial_guess = np.array([np.random.uniform(low, high) for low, high in bounds])

            remaining_budget -= result.nfev

        return best_solution, best_value