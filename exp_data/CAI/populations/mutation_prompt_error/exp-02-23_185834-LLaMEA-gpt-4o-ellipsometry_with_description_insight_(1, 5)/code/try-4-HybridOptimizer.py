import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Retrieve bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Adaptive Sampling
        initial_samples = min(self.budget // 3, 15 * self.dim)
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Find the best sample
        evaluations.sort(key=lambda x: x[0])
        best_sample = evaluations[0][1]
        
        # Step 2: Dynamic Local Optimization
        if self.evaluations < self.budget:
            local_budget = max(self.budget - self.evaluations, 5 * self.dim)
            options = {'maxiter': local_budget}
            result = minimize(func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
            best_sample = result.x
        
        return best_sample