import numpy as np
from scipy.optimize import minimize

class MultiStartLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_initial_samples = min(10 * self.dim, self.budget // (2 * self.dim))
        initial_samples = self._uniform_sampling(bounds, num_initial_samples)
        
        best_sample = None
        best_value = float('inf')
        self.budget -= num_initial_samples
        
        for sample in initial_samples:
            if self.budget <= 0:
                break

            res = minimize(func, sample, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget // len(initial_samples)})
            self.budget -= res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_sample = res.x

        return best_sample

    def _uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.array([np.random.uniform(low, high) for low, high in bounds])
            samples.append(sample)
        return samples