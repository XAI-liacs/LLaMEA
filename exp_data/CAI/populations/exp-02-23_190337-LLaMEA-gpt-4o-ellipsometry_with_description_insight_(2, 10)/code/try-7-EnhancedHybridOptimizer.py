import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
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
        
        # Allocate budget for initial sampling and optimization
        sample_budget = int(0.25 * self.budget)
        optimizer_budget = self.budget - sample_budget

        # Adaptive distribution for better initial sampling
        initial_guesses = np.random.normal(
            loc=(lb + ub) / 2, 
            scale=(ub - lb) / 4, 
            size=(sample_budget, self.dim)
        ).clip(lb, ub)

        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Dynamic refinement factor based on initial sampling variance
        refinement_factor = min(0.1, np.std(initial_guesses, axis=0) / (ub - lb))

        # Refine bounds using dynamic thresholding
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor[i] * (u - l)),
                min(u, best_solution[i] + refinement_factor[i] * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        # Define BFGS optimization function
        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun

        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution