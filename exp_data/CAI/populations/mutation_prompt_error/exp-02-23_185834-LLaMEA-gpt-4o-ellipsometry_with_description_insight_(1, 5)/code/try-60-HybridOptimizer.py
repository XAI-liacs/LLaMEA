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
        
        # Step 1: Adaptive Sampling with strategic refinement
        initial_samples = min(self.budget // 3, 50 * self.dim)
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
        top_samples = [e[1] for e in evaluations[:5]]
        
        # Apply a mutation mechanism to top candidates for exploration
        mutated_samples = top_samples + [s + np.random.normal(0, 0.1, self.dim) for s in top_samples]  # Change 1 & 2

        # Step 2: Dynamic boundary tightening and local L-BFGS-B optimization
        best_sample = None
        best_value = float('inf')
        for sample in mutated_samples:  # Change 3
            if self.evaluations < self.budget:
                local_budget = self.budget - self.evaluations
                options = {'maxiter': local_budget, 'gtol': 1e-9, 'ftol': 1e-9}
                result = minimize(func, sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=options)  # Change 4
                if result.success and result.fun < best_value:
                    best_sample = result.x
                    best_value = result.fun
                    if result.fun < 1e-7:
                        break
        
        return best_sample if best_sample is not None else top_samples[0]