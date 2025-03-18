import numpy as np
from scipy.optimize import minimize

class AdaptiveTwoPhaseOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Retrieve bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Step 1: Gaussian Sampling for Exploration
        initial_samples = min(self.budget // 3, 8 * self.dim)
        samples = np.random.normal(loc=(lb + ub) / 2, scale=(ub - lb) / 4, size=(initial_samples, self.dim))
        samples = np.clip(samples, lb, ub)
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

        # Step 2: Adaptive Local Optimization using Nelder-Mead
        if self.evaluations < self.budget:
            local_budget = self.budget - self.evaluations
            options = {'maxiter': local_budget, 'adaptive': True}
            result = minimize(func, best_sample, method='Nelder-Mead', options=options)
            best_sample = result.x

        return best_sample