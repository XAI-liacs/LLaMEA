import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        num_initial_samples = min(self.budget // 2, max(15, 2 * self.dim))
        
        # Use Latin Hypercube sampling for better initial coverage
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_samples = qmc.scale(sampler.random(n=num_initial_samples), lb, ub)
        
        best_solution = None
        best_value = float('inf')
        
        for sample in initial_samples:
            result = minimize(func, sample, method='L-BFGS-B', bounds=np.array(list(zip(lb, ub))))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        adaptive_shrink_rate = 0.05
        
        remaining_budget = self.budget - num_initial_samples
        while remaining_budget > 0:
            current_bounds = [(max(lb[i], best_solution[i] - adaptive_shrink_rate * (ub[i] - lb[i])),
                               min(ub[i], best_solution[i] + adaptive_shrink_rate * (ub[i] - lb[i]))) for i in range(self.dim)]
            
            # Introduce a small mutation for diversification
            mutation = np.random.normal(scale=0.01, size=self.dim)
            candidate_solution = best_solution + mutation
            candidate_solution = np.clip(candidate_solution, lb, ub)  # Ensure within bounds
            
            result = minimize(func, candidate_solution, method='L-BFGS-B', bounds=current_bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            adaptive_shrink_rate *= 0.9
            remaining_budget -= 1
        
        return best_solution