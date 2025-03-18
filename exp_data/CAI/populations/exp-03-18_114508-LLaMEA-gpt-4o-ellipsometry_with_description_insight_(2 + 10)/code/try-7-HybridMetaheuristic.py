import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.used_budget = 0

    def __call__(self, func):
        # Get initial bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Determine the number of initial samples based on the budget
        num_initial_samples = min(self.budget // 4, 15)  # Changed sampling strategy
        points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        # Evaluate all initial samples
        scores = [func(x) for x in points]
        self.used_budget += num_initial_samples
        
        # Find the best initial sample
        best_idx = np.argmin(scores)
        best_point = points[best_idx]
        best_score = scores[best_idx]

        # Run Nelder-Mead from this best initial point
        result = minimize(func, best_point, method='Nelder-Mead',
                          options={'maxiter': (self.budget - self.used_budget) // 2,  # Adjusted iterations
                                   'xatol': 1e-4, 'fatol': 1e-4})  # Tightened tolerances
        self.used_budget += result.nfev
        
        # If budget allows, refine the search bounds around the best solution found
        if self.used_budget < self.budget:
            refined_lb = np.maximum(lb, result.x - 0.05 * (ub - lb))  # Narrower refinement
            refined_ub = np.minimum(ub, result.x + 0.05 * (ub - lb))
            
            # Further optimize within refined bounds
            result_refined = minimize(func, result.x, method='L-BFGS-B', bounds=list(zip(refined_lb, refined_ub)),
                                      options={'maxiter': self.budget - self.used_budget})
            self.used_budget += result_refined.nfev
            
            if result_refined.fun < result.fun:
                return result_refined.x
        
        return result.x