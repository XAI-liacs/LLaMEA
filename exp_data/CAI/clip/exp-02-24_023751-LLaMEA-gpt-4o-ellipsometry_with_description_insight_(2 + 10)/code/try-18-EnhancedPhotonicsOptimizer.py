import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedPhotonicsOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Sobol sequence for low-discrepancy sampling
        sobol_sampler = qmc.Sobol(d=self.dim, scramble=True)
        initial_samples = lb + (ub - lb) * sobol_sampler.random(num_samples)
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Convert bounds to a format compatible with minimize
            bounds = [(lb[i], ub[i]) for i in range(self.dim)]
            
            # Local optimization using L-BFGS-B with constrained bounds
            result = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': min(self.budget - evaluations, 100)})
            
            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
                
            # Dynamically adjust bounds based on current best solution
            if best_solution is not None:
                lb = np.maximum(lb, best_solution - 0.05 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.05 * (ub - lb))
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution