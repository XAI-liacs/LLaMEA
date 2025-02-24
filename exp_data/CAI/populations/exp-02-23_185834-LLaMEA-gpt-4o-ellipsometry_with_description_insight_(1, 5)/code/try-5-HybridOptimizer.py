import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Retrieve bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Sobol Sequence for quasi-random initial sampling
        initial_samples = min(self.budget // 2, 10 * self.dim)
        sobol = Sobol(d=self.dim, scramble=True)
        samples = sobol.random(n=initial_samples)
        samples = lb + samples * (ub - lb)  # Scale samples to the bounds
        
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
        
        # Step 2: Local Optimization using BFGS
        if self.evaluations < self.budget:
            local_budget = self.budget - self.evaluations
            options = {'maxiter': local_budget}
            result = minimize(func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
            best_sample = result.x
        
        return best_sample