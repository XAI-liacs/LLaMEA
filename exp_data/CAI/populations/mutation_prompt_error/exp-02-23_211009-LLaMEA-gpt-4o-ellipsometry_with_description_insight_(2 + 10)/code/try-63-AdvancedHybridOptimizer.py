import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 10)  # Adjusted line
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points with dynamic sampling strategy
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        best_value = float('inf')
        best_solution = None
        
        # Evaluate sampled points
        evaluations = 0
        threshold = float('inf')  # Initialize a threshold for selective local refinement
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
                threshold = best_value * 1.05  # Set threshold to refine solutions close to the best

        # Selectively refine only those solutions within the adaptive threshold
        for sample in samples:
            if evaluations >= self.budget or func(sample) > threshold:
                continue
            result = minimize(func, sample, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options={'maxiter': remaining_budget, 'disp': False})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution