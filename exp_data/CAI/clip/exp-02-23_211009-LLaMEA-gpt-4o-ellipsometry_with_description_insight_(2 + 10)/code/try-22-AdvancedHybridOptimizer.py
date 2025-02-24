import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (3 * self.dim), 50), 15)  # Tweaked sampling strategy
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points with dynamic sampling strategy
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        # Evaluate sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Adaptive subspace search modification
        subspace_dim = max(1, self.dim // 2)
        subspace_indices = np.random.choice(self.dim, subspace_dim, replace=False)
        
        def bounded_func(x):
            x_full = np.copy(best_solution)
            x_full[subspace_indices] = x  # Modify only subspace dimensions
            return func(np.clip(x_full, lb, ub))
        
        # Use the remaining budget efficiently in local optimization with adaptive L-BFGS-B
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(bounded_func, best_solution[subspace_indices], method='L-BFGS-B', 
                          bounds=np.array([lb[subspace_indices], ub[subspace_indices]]).T, options=options)
        
        best_solution[subspace_indices] = result.x  # Update the best solution with optimized subspace
        return best_solution