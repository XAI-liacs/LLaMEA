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
        
        # Step 1: Adaptive Sampling with strategic refinement
        initial_samples = min(self.budget // 2, 100 * self.dim)  # Change 1
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(initial_samples))) * (ub - lb) + lb
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Sort and select top candidates for local optimization
        evaluations.sort(key=lambda x: x[0])
        top_samples = [e[1] for e in evaluations[:7]]  # Change 2: Increased top candidates
        
        # Step 2: Dynamic boundary tightening and local optimization
        best_sample = None
        best_value = float('inf')
        for sample in top_samples:
            if self.evaluations < self.budget:
                local_budget = self.budget - self.evaluations
                options = {'maxiter': local_budget, 'xatol': 1e-10, 'fatol': 1e-10}  # Change 3 (tightened tolerances)
                result = minimize(func, sample, method='Nelder-Mead', options=options)
                if result.success and result.fun < best_value:
                    best_sample = result.x
                    best_value = result.fun
                    if result.fun < 1e-8:  # Change 4
                        break
        
        return best_sample if best_sample is not None else top_samples[0]