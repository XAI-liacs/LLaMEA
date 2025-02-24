import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Adaptive Sampling to explore
        initial_samples = min(self.budget // 2, 20 * self.dim)  # Increased adaptive sampling
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        evaluations.sort(key=lambda x: x[0])
        best_sample = evaluations[0][1]
        
        # Step 2: Local Optimization with gradient descent and momentum
        if self.evaluations < self.budget:
            local_budget = self.budget - self.evaluations
            options = {'maxiter': local_budget}
            result = minimize(func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=options)
            if result.success:
                best_sample = result.x
        
        return best_sample