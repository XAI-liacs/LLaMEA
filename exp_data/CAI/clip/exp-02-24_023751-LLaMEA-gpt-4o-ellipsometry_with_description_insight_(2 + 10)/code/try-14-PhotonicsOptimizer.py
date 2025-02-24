import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class PhotonicsOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Generate initial samples using Sobol sequence
        sobol = Sobol(d=self.dim, scramble=True)
        initial_samples = lb + (ub - lb) * sobol.random_base2(m=int(np.log2(num_samples)))
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Convert bounds to a format compatible with minimize
            bounds = [(lb[i], ub[i]) for i in range(self.dim)]
            
            # Local optimization using a dynamically adjusted Nelder-Mead
            result = minimize(func, sample, method='Nelder-Mead', bounds=bounds, options={'maxfev': min(self.budget - evaluations, 100)})
            
            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
                
            # Dynamically adjust bounds based on current best solution
            if best_solution is not None:
                lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution