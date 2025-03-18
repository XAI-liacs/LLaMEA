import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initial sampling setup
        initial_samples = min(self.budget // 10, 100)
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        # Evaluate initial sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
                
        # Adaptive resampling in promising regions
        refined_samples = min(max(remaining_budget // 2, 5), 50)
        local_region = 0.2 * (ub - lb)  # 20% of the range
        adaptive_lb = np.clip(best_solution - local_region, lb, ub)
        adaptive_ub = np.clip(best_solution + local_region, lb, ub)
        adaptive_samples = np.random.uniform(adaptive_lb, adaptive_ub, (refined_samples, self.dim))
        
        # Evaluate refined samples
        for sample in adaptive_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Use local optimization with BFGS starting from the best adaptive point
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Use up remaining budget in local optimization
        local_budget = self.budget - evaluations
        options = {'maxiter': local_budget * 1.2, 'disp': False}  # Increased emphasis on local optimization
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        return result.x