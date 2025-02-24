import numpy as np
from scipy.optimize import minimize
from pyDOE2 import lhs

class HybridSamplingLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Generate initial samples using Latin Hypercube Sampling
        lhs_samples = lhs(self.dim, samples=num_samples)
        initial_samples = lb + (ub - lb) * lhs_samples
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Local optimization using Sequential Quadratic Programming (SQP)
            result = minimize(func, sample, method='SLSQP', bounds=zip(lb, ub), options={'maxiter': self.budget - evaluations})
            
            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution