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
        
        # Dynamic budget allocation
        initial_sample_budget = max(10, int(0.15 * self.budget))
        exploration_budget = int(0.15 * self.budget)  # Adjusted exploration budget
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget

        # Initial uniform sampling for initial guesses
        initial_guesses = np.random.uniform(lb * 0.95, ub * 1.05, (initial_sample_budget, self.dim))

        # Evaluate initial guesses
        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Enhanced exploration phase with adaptive variance
        exploration_variance = np.std(initial_guesses, axis=0) * 0.95  # Updated exploration variance
        exploration_guesses = best_solution + np.random.normal(0, exploration_variance, (exploration_budget, self.dim))
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Refine bounds using gradient information
        refinement_factor_max = 0.08  # Adjusted refinement factor
        refinement_factor = min(refinement_factor_max, 0.1 * np.linalg.norm(exploration_variance))
        gradient = np.gradient(func(np.array([best_solution])))[0]  # Estimate gradient
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * gradient[i] * (u - l)),
                min(u, best_solution[i] + refinement_factor * gradient[i] * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        # Define the BFGS optimization function
        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget, 'learning_rate': 0.1})
            return res.x, res.fun

        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution