import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        # Get bounds from func
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        # Dynamically adjust the number of initial samples based on remaining budget
        num_initial_samples = min(20, self.budget // 8)  # Changed from 16 to 20
                
        # Generate initial samples by uniform sampling
        initial_samples = np.random.uniform(lb, ub, size=(num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            # Add random perturbation inspired by simulated annealing
            perturbation = np.random.normal(0, 0.01, size=self.dim)
            perturbed_sample = sample + perturbation
            
            # Local optimization using L-BFGS-B starting from each perturbed sample
            res = minimize(func, perturbed_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            
            # Update best solution found so far
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            # Iteratively adjust bounds based on the best solution found
            radius = 0.05 * (ub - lb)
            lb = np.maximum(lb, best_solution - radius)
            ub = np.minimum(ub, best_solution + radius)
            bounds = list(zip(lb, ub))
        
        return best_solution