import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class ImprovedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        num_samples = min(max(7, int(self.budget * 0.4)), self.budget // 2)
        
        # Step 1: Sobol sequence for initial exploration
        sampler = Sobol(d=self.dim, scramble=True)
        initial_samples = sampler.random_base2(m=int(np.log2(num_samples))) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best sample and refine using trust-region method
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]

        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]

        # Use trust-region method for local optimization
        result = minimize(
            func,
            best_guess,
            method='trust-constr',
            bounds=local_bounds,
            options={'maxiter': self.budget - self.evals}
        )
        
        self.evals += result.nfev

        # Step 3: Progressive tightening of bounds
        while self.evals < self.budget:
            scale_factor = max(0.05, 0.15 * (1 - self.evals / self.budget))
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]

            result = minimize(
                func,
                result.x,
                method='trust-constr',
                bounds=narrowed_bounds,
                options={'maxiter': self.budget - self.evals}
            )
            
            self.evals += result.nfev
        
        return result.x