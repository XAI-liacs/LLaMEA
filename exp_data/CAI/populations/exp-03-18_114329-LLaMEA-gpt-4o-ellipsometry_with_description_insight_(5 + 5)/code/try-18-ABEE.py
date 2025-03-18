import numpy as np
from scipy.optimize import minimize

class ABEE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget
        
        # Initialize with uniform random sampling
        best_solution = None
        best_value = np.inf
        
        for _ in range(remaining_budget // 4):
            x0 = np.random.uniform(lb, ub)
            value = func(x0)
            remaining_budget -= 1
            
            if value < best_value:
                best_value = value
                best_solution = x0

            if remaining_budget <= 0:
                break
        
        # Local optimization using successful initial guesses
        def bounded_minimize(x0):
            nonlocal remaining_budget
            result = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            remaining_budget -= result.nfev
            return result.x, result.fun, remaining_budget
        
        if best_solution is not None:
            best_solution, best_value, remaining_budget = bounded_minimize(best_solution)
        
        # Iteratively refine bounds and re-optimize
        while remaining_budget > 0:
            # Shrink bounds based on the last best solution
            new_lb = np.maximum(lb, best_solution - (ub - lb) * 0.1)  # Reduced bound adjustment factor
            new_ub = np.minimum(ub, best_solution + (ub - lb) * 0.1)  # Reduced bound adjustment factor
            
            x0 = np.random.uniform(new_lb, new_ub)
            local_solution, local_value = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(new_lb, new_ub))).x, func(x0)
            remaining_budget -= 1
            
            if local_value < best_value:
                best_value = local_value
                best_solution = local_solution
                
            if remaining_budget <= 0:
                break
            
        return best_solution