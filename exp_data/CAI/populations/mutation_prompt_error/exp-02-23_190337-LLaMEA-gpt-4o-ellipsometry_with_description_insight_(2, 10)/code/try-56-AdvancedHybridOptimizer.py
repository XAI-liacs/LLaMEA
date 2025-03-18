import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
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
        
        # Dynamic budget allocation
        initial_sample_budget = max(10, int(0.15 * self.budget))
        exploration_budget = int(0.1 * self.budget)
        intermediate_budget = int(0.1 * self.budget)  # New intermediate phase
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget - intermediate_budget

        # Initial uniform sampling for initial guesses
        initial_guesses = np.random.uniform(lb * 0.95, ub * 1.05, (initial_sample_budget, self.dim))

        # Evaluate initial guesses
        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Exploration phase: perturb best guess to sample nearby regions
        exploration_variance = np.var(initial_guesses, axis=0) * 1.1 + 0.01  # Adjusted variance scaling
        exploration_guesses = best_solution + np.random.uniform(-0.15, 0.15, (exploration_budget, self.dim)) * exploration_variance
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Intermediate local optimization using Nelder-Mead
        def intermediate_optimization(x0):
            res = minimize(func, x0, method='Nelder-Mead', options={'maxfev': intermediate_budget})
            return res.x, res.fun
        
        best_solution, best_value = intermediate_optimization(best_solution)

        # Refine bounds to local region around the best guess
        refinement_factor_max = 0.15
        refinement_factor = min(refinement_factor_max, 0.2 * np.mean(exploration_variance))
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