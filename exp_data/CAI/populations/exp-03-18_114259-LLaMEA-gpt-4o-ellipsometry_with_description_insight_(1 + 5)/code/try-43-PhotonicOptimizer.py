import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        # Hybrid initial sampling strategy: Combine uniform and Latin Hypercube sampling
        num_initial_samples = min(20, self.budget // 8)  # Increased initial samples
        initial_samples_uniform = np.random.uniform(lb, ub, size=(num_initial_samples // 2, self.dim))
        initial_samples_lhs = np.random.uniform(lb, ub, size=(num_initial_samples // 2, self.dim))
        initial_samples = np.vstack((initial_samples_uniform, initial_samples_lhs))
        
        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            # Local optimization using L-BFGS-B starting from each sample
            res = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            
            # Update best solution found so far
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            # More adaptive bounds reduction based on exploration feedback
            if i % 2 == 0:  # Adjust bounds only on every other sample
                radius = 0.03 * (ub - lb)  # Reduced radius for finer exploration
                lb = np.maximum(lb, best_solution - radius)
                ub = np.minimum(ub, best_solution + radius)
                bounds = list(zip(lb, ub))
        
        return best_solution