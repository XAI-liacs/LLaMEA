import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        bounds = np.array(list(zip(lb, ub)))

        best_solution = None
        best_value = float('inf')
        
        initial_sample_budget = max(10, int(0.15 * self.budget))
        exploration_budget = int(0.1 * self.budget)
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget

        initial_guesses = np.random.uniform(lb * 0.95, ub * 1.05, (initial_sample_budget, self.dim))

        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        exploration_variance = np.std(initial_guesses, axis=0) * 0.85  # Adjusted exploration variance
        exploration_guesses = best_solution + np.random.normal(0, exploration_variance, (exploration_budget, self.dim))  # Changed to Gaussian exploration
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        refinement_factor_max = 0.08  # Reduced from 0.1 to 0.08 for finer local search
        refinement_factor = min(refinement_factor_max, 0.2 * np.linalg.norm(exploration_variance))
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun

        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution