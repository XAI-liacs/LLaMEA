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
        
        # Step 1: Uniform Sampling to explore
        initial_samples = min(self.budget // 2 + 1, 25 * self.dim)  # Increased initial sampling
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Sort samples by evaluation result
        evaluations.sort(key=lambda x: x[0])
        
        # Step 2: Local Optimization using BFGS with safeguarded starting point
        if self.evaluations < self.budget:
            for i in range(min(3, len(evaluations))):  # Use top 3 samples for better local search
                best_sample = evaluations[i][1]
                local_budget = self.budget - self.evaluations
                options = {'maxiter': local_budget, 'gtol': 1e-6}  # Enhanced convergence precision
                result = minimize(func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
                if result.success:  # safeguard to ensure convergence
                    best_sample = result.x
                    break
        
        return best_sample