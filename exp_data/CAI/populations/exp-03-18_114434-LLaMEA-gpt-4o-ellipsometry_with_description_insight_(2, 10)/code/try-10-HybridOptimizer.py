import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Step 1: Uniform Sampling for Initial Guess
        num_initial_points = max(1, self.budget // 10)  # Allocate 10% of the budget for initial exploration
        initial_guesses = []
        for _ in range(num_initial_points):
            guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            initial_guesses.append((func(guess), guess))

        # Select the best initial guess
        initial_guesses.sort(key=lambda x: x[0])
        best_initial_value, best_initial_guess = initial_guesses[0]

        # Step 2: Local Optimization using BFGS
        remaining_budget = self.budget - num_initial_points
        if remaining_budget > 0:
            result = minimize(
                func, 
                best_initial_guess, 
                method='BFGS',
                options={'maxiter': remaining_budget}
            )
            return result.x if result.success else best_initial_guess
        else:
            return best_initial_guess