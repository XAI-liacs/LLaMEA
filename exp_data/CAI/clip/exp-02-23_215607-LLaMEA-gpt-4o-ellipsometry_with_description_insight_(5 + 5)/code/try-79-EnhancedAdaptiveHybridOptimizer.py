import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Improved initial sampling using Sobol sequence for better coverage
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_initial_samples = min(15, self.budget // 3)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        initial_samples = qmc.scale(sampler.random_base2(m=int(np.log2(num_initial_samples))), bounds[:, 0], bounds[:, 1])
        
        evals = 0
        best_sample = None
        best_value = float('inf')

        # Evaluate initial samples
        for sample in initial_samples:
            if evals >= self.budget:
                break
            value = func(sample)
            evals += 1
            if value < best_value:
                best_value = value
                best_sample = sample

        # Local optimization using trust-region method with refined step-size control
        if evals < self.budget:
            def wrapped_func(x):
                nonlocal evals
                if evals >= self.budget or best_value == 0:
                    return float('inf')
                value = func(x)
                evals += 1
                return value

            result = minimize(
                wrapped_func, 
                best_sample, 
                method='trust-constr', 
                bounds=bounds, 
                options={'maxiter': self.budget - evals}
            )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

        return best_sample