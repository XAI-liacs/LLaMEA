import numpy as np
from scipy.optimize import minimize, Bounds

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize variables
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Uniform random sampling for initialization with a larger initial sample size
        initial_guesses = [np.random.uniform(lb, ub) for _ in range(min(10, self.budget))]
        
        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            # Step 2: Apply Nelder-Mead for initial local optimization
            result_nm = minimize(func, guess, method='Nelder-Mead', options={'maxfev': (self.budget - evaluations)})
            evaluations += result_nm.nfev
            if result_nm.fun < best_value:
                best_value = result_nm.fun
                best_solution = result_nm.x
            
            # Step 3: Refine with BFGS within dynamic bounds
            if evaluations < self.budget:
                # Correct dynamic bounds logic
                lower_dynamic_bound = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                upper_dynamic_bound = np.minimum(ub, best_solution + 0.1 * (ub - lb))
                current_bounds = Bounds(lower_dynamic_bound, upper_dynamic_bound)
                result_bfgs = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds, options={'maxfun': (self.budget - evaluations)})
                evaluations += result_bfgs.nfev
                if result_bfgs.fun < best_value:
                    best_value = result_bfgs.fun
                    best_solution = result_bfgs.x
        
        return best_solution