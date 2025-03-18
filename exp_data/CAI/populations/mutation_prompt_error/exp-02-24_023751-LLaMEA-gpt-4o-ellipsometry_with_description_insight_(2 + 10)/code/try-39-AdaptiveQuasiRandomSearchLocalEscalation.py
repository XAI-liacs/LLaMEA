import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from scipy.stats import qmc

class AdaptiveQuasiRandomSearchLocalEscalation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize Halton sequence sampler
        sampler = qmc.Halton(d=self.dim, scramble=True)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Generate initial Halton samples
        initial_samples = qmc.scale(sampler.random(num_samples), lb, ub)
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Local optimization using Nelder-Mead
            result = minimize(func, sample, method='Nelder-Mead', options={'maxfev': self.budget - evaluations, 'adaptive': True})

            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
                
                # Adjust simplex size based on the improvement
                simplex_size_factor = 0.05 if result.fun < 1e-3 else 0.1
                lb = np.maximum(lb, best_solution - simplex_size_factor * (ub - lb))
                ub = np.minimum(ub, best_solution + simplex_size_factor * (ub - lb))
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution