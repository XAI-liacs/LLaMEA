import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize  # Importing Bayesian optimization

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
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget

        # Initial sampling using Bayesian optimization
        def objective(x):
            return func(np.array(x))
        
        gp_result = gp_minimize(objective, bounds, n_calls=initial_sample_budget, random_state=0)
        initial_guesses = np.array(gp_result.x_iters)
        
        # Evaluate initial guesses
        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Exploration phase: perturb best guess to sample nearby regions
        exploration_variance = np.var(initial_guesses, axis=0)  # Calculate variance
        exploration_guesses = best_solution + np.random.uniform(-0.1, 0.1, (exploration_budget, self.dim)) * (1 + exploration_variance * 1.5) 
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Refine bounds to local region around the best guess
        refinement_factor_max = 0.15  
        refinement_factor = min(refinement_factor_max, 0.2 * np.linalg.norm(exploration_variance))
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        # Define the BFGS optimization function
        def bfgs_optimization(x0):
            learning_rate = 0.05 / (1 + exploration_variance.mean())  # More refined adaptive learning rate
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun

        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution