import numpy as np
from scipy.optimize import minimize

class PhotonicsOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Generate initial random samples
        initial_samples = lb + (ub - lb) * np.random.rand(num_samples, self.dim)
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Local optimization using Nelder-Mead
            result = minimize(func, sample, method='Nelder-Mead', bounds=func.bounds, options={'maxfev': self.budget - evaluations})
            
            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution