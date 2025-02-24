import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

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
        
        # Initial uniform sampling for initial guesses
        initial_guesses = np.random.uniform(lb, ub, (initial_sample_budget, self.dim))

        # Evaluate initial guesses
        initial_values = []
        for guess in initial_guesses:
            value = func(guess)
            initial_values.append(value)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Calculate variance and use KD-Tree for neighborhood exploration
        exploration_variance = np.var(initial_guesses, axis=0)
        tree = KDTree(initial_guesses)
        neighbors_idx = tree.query_ball_point(best_solution, r=0.1)
        
        # Exploration phase: perturb best guess using neighbors
        exploration_guesses = best_solution + np.random.uniform(-0.05, 0.05, (exploration_budget, self.dim)) * (1 + exploration_variance)
        exploration_guesses = np.clip(exploration_guesses, lb, ub)
        
        for guess in exploration_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        # Adaptive bound shrinking based on convergence trends
        convergence_trend = np.mean(initial_values) - best_value
        refinement_factor_max = 0.2 * np.exp(-0.5 * convergence_trend)
        refinement_factor = min(refinement_factor_max, 0.1 * best_value)
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])
        
        # Define the BFGS optimization function with local surrogate
        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun
        
        # Execute BFGS optimization from the best initial guess
        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution