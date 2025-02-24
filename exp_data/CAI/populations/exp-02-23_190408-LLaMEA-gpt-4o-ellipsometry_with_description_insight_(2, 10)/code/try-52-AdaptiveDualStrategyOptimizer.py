import numpy as np
from scipy.optimize import minimize

class AdaptiveDualStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        grid_sampling_budget = min(5 * self.dim, self.budget // 3)
        grid_samples = self.grid_sampling(bounds, grid_sampling_budget)
        
        best_sample = None
        best_value = float('inf')
        
        for sample in grid_samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample
        
        remaining_budget = self.budget - grid_sampling_budget
        
        res = self.local_optimization(func, best_sample, bounds, remaining_budget)
        
        return res.x, res.fun

    def grid_sampling(self, bounds, num_samples):
        grid_points_per_dim = int(num_samples ** (1 / self.dim))
        linspaces = [np.linspace(low, high, grid_points_per_dim) for low, high in bounds]
        grid_samples = np.array(np.meshgrid(*linspaces)).T.reshape(-1, self.dim)
        return grid_samples

    def local_optimization(self, func, initial_guess, bounds, budget):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': budget})
        return res