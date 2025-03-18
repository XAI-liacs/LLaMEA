import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.used_budget = 0

    def __call__(self, func):
        # Get initial bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Use Sobol sequence for initial sampling
        sampler = Sobol(d=self.dim, scramble=True)
        num_initial_samples = min(self.budget // 3, 10)
        points = sampler.random_base2(m=int(np.log2(num_initial_samples)))
        points = lb + points * (ub - lb)
        
        # Evaluate all initial samples
        scores = [func(x) for x in points]
        self.used_budget += num_initial_samples
        
        # Find the best initial sample
        best_idx = np.argmin(scores)
        best_point = points[best_idx]
        best_score = scores[best_idx]

        # Use L-BFGS-B for better handling of bounds
        result = minimize(func, best_point, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                          options={'maxiter': self.budget - self.used_budget})
        self.used_budget += result.nfev
        
        # If budget allows, refine the search bounds around the best solution found
        if self.used_budget < self.budget:
            refined_lb = np.maximum(lb, result.x - 0.1 * (ub - lb))
            refined_ub = np.minimum(ub, result.x + 0.1 * (ub - lb))
            
            # Further optimize within refined bounds
            result_refined = minimize(func, result.x, method='L-BFGS-B', bounds=list(zip(refined_lb, refined_ub)),
                                      options={'maxiter': self.budget - self.used_budget})
            self.used_budget += result_refined.nfev
            
            if result_refined.fun < result.fun:
                return result_refined.x
        
        return result.x