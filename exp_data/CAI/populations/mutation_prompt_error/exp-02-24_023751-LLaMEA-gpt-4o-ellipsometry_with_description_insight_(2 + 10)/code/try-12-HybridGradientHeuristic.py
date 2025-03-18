import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from skopt import gp_minimize  # Added line for Bayesian optimization

class HybridGradientHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        
        num_samples = max(5, self.budget // 10)
        
        initial_samples = qmc.scale(sampler.random_base2(m=int(np.log2(num_samples))), lb, ub)
        
        best_solution = None
        best_score = float('inf')
        
        evaluations = 0

        for sample in initial_samples:
            # Local optimization using L-BFGS-B
            result = minimize(func, sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': self.budget - evaluations})
            
            evaluations += result.nfev
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
                lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            
            if evaluations >= self.budget:
                break
        
        if evaluations < self.budget:  # Added line to use Bayesian optimization
            result = gp_minimize(func, list(zip(lb, ub)), n_calls=self.budget-evaluations, x0=best_solution)
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        
        return best_solution