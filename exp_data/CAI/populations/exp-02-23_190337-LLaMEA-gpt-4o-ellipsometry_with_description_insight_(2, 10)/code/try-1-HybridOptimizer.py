import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        bounds = np.array(list(zip(lb, ub)))

        # Initialize best solution variables
        best_solution = None
        best_value = float('inf')
        
        # Divide budget: 20% for initial sampling, 80% for BFGS optimization
        sample_budget = int(0.2 * self.budget)
        optimizer_budget = self.budget - sample_budget

        # Initial uniform sampling for initial guesses
        initial_guesses = np.random.uniform(lb, ub, (sample_budget, self.dim))

        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Refine bounds to local region around the best initial guess
        refinement_factor = 0.1  # 10% of the range around the best guess
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        # Define the BFGS optimization function
        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun

        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution