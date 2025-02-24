import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _uniform_sample(self, bounds, num_samples):
        return np.array([np.random.uniform(low=bounds.lb[i], high=bounds.ub[i], size=num_samples) for i in range(self.dim)]).T

    def _bounded_minimize(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    def __call__(self, func):
        # Initial uniform sampling
        num_initial_samples = 5
        samples = self._uniform_sample(func.bounds, num_initial_samples)
        evaluations = []
        
        for sample in samples:
            if len(evaluations) >= self.budget:
                break
            res = minimize(func, sample, method='Nelder-Mead', options={'maxfev': self.budget - len(evaluations)})
            evaluations.extend([res.nfev])  # Fixed: ensure evaluations is extended with a list
            x0 = res.x
        
        # Main optimization loop
        remaining_budget = self.budget - len(evaluations)
        
        if remaining_budget > 0:
            for _ in range(remaining_budget):
                bounds = [(max(func.bounds.lb[i], x0[i] - 0.1), min(func.bounds.ub[i], x0[i] + 0.1)) for i in range(self.dim)]
                x0, _ = self._bounded_minimize(func, x0, bounds)
        
        return x0