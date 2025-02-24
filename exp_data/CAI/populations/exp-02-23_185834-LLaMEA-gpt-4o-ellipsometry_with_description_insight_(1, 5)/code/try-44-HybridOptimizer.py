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
        
        # Step 1: Uniform Sampling with strategic refinement
        initial_samples = min(self.budget // 2.5, 25 * self.dim)  # Change 1
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Sort and select top candidates for local optimization
        evaluations.sort(key=lambda x: x[0])
        top_samples = [e[1] for e in evaluations[:6]]  # Change 2
        
        # Step 2: Dynamic boundary tightening and local BFGS optimization
        best_sample = None
        best_value = float('inf')
        for sample in top_samples:
            if self.evaluations < self.budget:
                local_budget = self.budget - self.evaluations
                options = {'maxiter': local_budget * 0.8, 'gtol': 1e-8}  # Change 3
                result = minimize(func, sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
                if result.success and result.fun < best_value:
                    best_sample = result.x
                    best_value = result.fun
                    if best_value < 1e-6:
                        break
        
        return best_sample if best_sample is not None else top_samples[0]