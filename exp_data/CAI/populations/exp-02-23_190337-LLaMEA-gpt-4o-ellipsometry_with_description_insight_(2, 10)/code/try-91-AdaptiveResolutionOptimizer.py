import numpy as np
from scipy.optimize import minimize

class AdaptiveResolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        bounds = np.array(list(zip(lb, ub)))

        # Initialize best solution variables
        best_solution = None
        best_value = float('inf')
        
        # Dynamic budget allocation
        initial_sample_budget = max(10, int(0.2 * self.budget))
        coarse_exploration_budget = int(0.3 * self.budget)
        fine_exploration_budget = int(0.2 * self.budget)
        optimizer_budget = self.budget - initial_sample_budget - coarse_exploration_budget - fine_exploration_budget

        # Initial uniform sampling for initial guesses
        initial_guesses = np.random.uniform(lb * 0.9, ub * 1.1, (initial_sample_budget, self.dim))  # Wider initial sampling range

        # Evaluate initial guesses
        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Coarse exploration phase: larger perturbations to explore broader regions
        coarse_variance = np.std(initial_guesses, axis=0) * 1.5
        coarse_guesses = best_solution + np.random.uniform(-0.5, 0.5, (coarse_exploration_budget, self.dim)) * (1 + coarse_variance)
        coarse_guesses = np.clip(coarse_guesses, lb, ub)
        
        for guess in coarse_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Fine exploration phase: smaller perturbations focused around the current best solution
        fine_variance = np.std(coarse_guesses, axis=0) * 0.3
        fine_guesses = best_solution + np.random.uniform(-0.1, 0.1, (fine_exploration_budget, self.dim)) * (1 + fine_variance)
        fine_guesses = np.clip(fine_guesses, lb, ub)
        
        for guess in fine_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Refine bounds to a narrower region around the best guess
        refinement_factor = 0.05  # Further narrowed down exploration
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        # Define the BFGS optimization function
        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget, 'learning_rate': 0.05})
            return res.x, res.fun

        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution