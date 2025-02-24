import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Step 1: Uniform Sampling for Initialization
        initial_points = np.random.uniform(lb, ub, (self.dim, self.dim))
        for point in initial_points:
            if self.evaluations >= self.budget:
                break
            value = func(point)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = point
        
        # Step 2: Local Search using Nelder-Mead
        options = {'maxiter': self.budget - self.evaluations, 'disp': False}
        result = minimize(func, best_solution, method='Nelder-Mead', options=options, bounds=[(l, u) for l, u in zip(lb, ub)])
        self.evaluations += result.nfev
        
        # Step 3: Adaptive Bounds Tightening
        if result.success:
            for i in range(self.dim):
                lb[i] = max(lb[i], result.x[i] - 0.1 * (ub[i] - lb[i]))
                ub[i] = min(ub[i], result.x[i] + 0.1 * (ub[i] - lb[i]))
            options['maxiter'] = self.budget - self.evaluations
            result = minimize(func, result.x, method='Nelder-Mead', options=options, bounds=[(l, u) for l, u in zip(lb, ub)])
            self.evaluations += result.nfev

        return result.x