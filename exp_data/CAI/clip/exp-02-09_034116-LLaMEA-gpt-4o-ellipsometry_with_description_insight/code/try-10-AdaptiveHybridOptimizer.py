import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Adaptive exploration phase: Sobol sequence sampling with increasing resolution
        num_initial_samples = max(1, self.budget // 20)  # Use 5% of budget initially
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        samples = sobol_sampler.random(num_initial_samples)
        evaluations_used = 0

        while evaluations_used + len(samples) < self.budget // 2:  # Use up to 50% of budget for sampling
            for sample in samples:
                sample_scaled = lb + sample * (ub - lb)
                value = func(sample_scaled)
                evaluations_used += 1
                if value < best_value:
                    best_value = value
                    best_solution = sample_scaled

            # Increase sampling resolution gradually
            remaining_budget = self.budget // 2 - evaluations_used
            new_samples_count = min(len(samples) * 2, remaining_budget)  # Double the samples or fit remaining budget
            samples = sobol_sampler.random(new_samples_count)

        # Dynamic exploitation phase: Local optimization using L-BFGS-B
        def wrapped_func(x):
            return func(x)
        
        minimizer_options = {'maxiter': self.budget - evaluations_used, 'disp': False}
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=minimizer_options)
        
        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x
        
        return best_solution