import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        # Adaptive initial exploration
        samples = np.random.uniform(lb, ub, (self.budget // 3, self.dim))
        adaptive_samples = np.mean(samples, axis=0) + np.random.normal(0, 0.1, (self.budget // 3, self.dim))
        samples = np.vstack((samples, adaptive_samples))
        
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value
            value = func(x)
            evaluations += 1
            return value

        dynamic_bounds = [(max(lb[i], best_solution[i] - 0.1), min(ub[i], best_solution[i] + 0.1)) for i in range(self.dim)]
        
        # Dual local optimization with L-BFGS-B and Nelder-Mead
        result_bfgs = minimize(local_optimization, best_solution, method='L-BFGS-B',
                               bounds=dynamic_bounds, options={'maxiter': (self.budget - evaluations) // 2, 'disp': False})
        
        if result_bfgs.fun < best_value:
            best_value = result_bfgs.fun
            best_solution = result_bfgs.x

        result_nm = minimize(local_optimization, best_solution, method='Nelder-Mead',
                             options={'maxiter': self.budget - evaluations, 'disp': False})
        
        if result_nm.fun < best_value:
            best_value = result_nm.fun
            best_solution = result_nm.x

        return best_solution, best_value