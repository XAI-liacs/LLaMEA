import numpy as np
from scipy.optimize import minimize

class VarianceGuidedLocalSearch:
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
        
        # Budget allocation
        initial_sample_budget = max(10, int(0.2 * self.budget))
        exploration_budget = int(0.2 * self.budget)
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget

        # Initial uniform sampling for initial guesses
        initial_guesses = np.random.uniform(lb, ub, (initial_sample_budget, self.dim))

        # Evaluate initial guesses
        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Exploration phase: variance-guided sampling
        exploration_variance = np.var(initial_guesses, axis=0)
        exploration_guesses = best_solution + np.random.normal(0, 0.1, (exploration_budget, self.dim)) * (1 + exploration_variance)
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Refine bounds and use variance to set dynamic learning rates
        refinement_factor = np.maximum(0.1, 0.1 * exploration_variance)
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor[i] * (u - l)),
                min(u, best_solution[i] + refinement_factor[i] * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        # Define the BFGS optimization function with dynamic learning rate
        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun

        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution