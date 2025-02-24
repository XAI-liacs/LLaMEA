import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryContractionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        best_solution = None
        best_value = float('inf')
        
        sample_budget = max(1, self.budget // 10)
        optimize_budget = self.budget - sample_budget
        contraction_factor = 0.8  # Initial contraction factor for narrowing bounds

        samples = np.random.uniform(lb, ub, (sample_budget, self.dim))

        for sample in samples:
            # Run local optimization with Nelder-Mead
            result = minimize(func, sample, method='Nelder-Mead', options={'maxiter': max(1, optimize_budget // sample_budget)})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Constrict bounds around the best solution found
                lb = np.maximum(lb, best_solution - contraction_factor * (ub - lb))
                ub = np.minimum(ub, best_solution + contraction_factor * (ub - lb))

        # Final refinement within constricted bounds
        refined_result = minimize(func, best_solution, method='Nelder-Mead', bounds=[(l, u) for l, u in zip(lb, ub)])
        if refined_result.fun < best_value:
            best_value = refined_result.fun
            best_solution = refined_result.x

        return best_solution