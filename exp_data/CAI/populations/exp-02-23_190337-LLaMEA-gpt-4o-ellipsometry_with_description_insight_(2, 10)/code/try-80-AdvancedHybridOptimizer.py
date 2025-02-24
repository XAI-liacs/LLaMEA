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
        
        initial_sample_budget = max(10, int(0.2 * self.budget))  # Adjusted sampling budget
        exploration_budget = int(0.15 * self.budget)  # Adjusted exploration budget
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget

        initial_guesses = np.random.uniform(lb * 0.95, ub * 1.05, (initial_sample_budget, self.dim))

        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        exploration_variance = np.var(initial_guesses, axis=0)
        exploration_guesses = best_solution + np.random.uniform(-0.1, 0.1, (exploration_budget, self.dim)) * (1 + exploration_variance * 2)  # Increased exploration scaling
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        refinement_factor_max = 0.20  # Adjusted refinement factor max
        refinement_factor = min(refinement_factor_max, 0.25 * np.linalg.norm(exploration_variance))  # Adjusted refinement factor calculation
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        def bfgs_optimization(x0):
            learning_rate = 0.1 / (1 + exploration_variance.mean())
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget, 'learning_rate': learning_rate})
            return res.x, res.fun

        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution