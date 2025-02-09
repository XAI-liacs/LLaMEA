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

        # Exploration phase: Sobol sequence sampling with dynamic adjustment
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        num_samples = max(1, int(np.sqrt(self.budget)))
        samples = sobol_sampler.random(num_samples)
        top_k = min(len(samples), int(self.budget * 0.1))  # Increased top-k percentage for better diversity
        top_samples = sorted(samples[:top_k], key=lambda s: func(lb + s * (ub - lb)))

        # Exploitation phase: Local optimization with L-BFGS-B and Trust-Region refinement
        for sample in top_samples:
            sample_scaled = lb + sample * (ub - lb)
            def wrapped_func(x):
                return func(x)

            # Initial local optimization with L-BFGS-B
            result = minimize(wrapped_func, sample_scaled, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': (self.budget // top_k) - num_samples})

            # Trust-region refinement for improved convergence
            if result.success:
                result = minimize(wrapped_func, result.x, method='trust-constr', bounds=list(zip(lb, ub)), options={'maxiter': (self.budget // top_k)})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution