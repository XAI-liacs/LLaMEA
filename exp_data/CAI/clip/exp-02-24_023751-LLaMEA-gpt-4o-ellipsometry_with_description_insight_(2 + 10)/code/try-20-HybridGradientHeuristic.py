import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridGradientHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize Sobol sequence sampler
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Generate initial Sobol samples
        initial_samples = qmc.scale(sampler.random_base2(m=int(np.log2(num_samples))), lb, ub)
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Local optimization using L-BFGS-B
            result = minimize(func, sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': self.budget - evaluations})
            
            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
                # Dynamically adjust bounds around the current best solution
                lb = np.maximum(lb, best_solution - 0.2 * (ub - lb))  # Changed from 0.1 to 0.2
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution