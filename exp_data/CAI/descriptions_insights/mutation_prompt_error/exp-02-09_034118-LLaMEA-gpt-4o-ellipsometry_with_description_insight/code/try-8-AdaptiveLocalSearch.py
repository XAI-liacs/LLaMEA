import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial Sobol sequence sampling
        sobol_sampler = Sobol(self.dim, scramble=True)
        num_initial_samples = min(5, self.budget // 2)
        samples = sobol_sampler.random_base2(m=int(np.log2(num_initial_samples)))
        samples = bounds[0] + samples * (bounds[1] - bounds[0])
        
        for x0 in samples:
            value = func(x0)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

        # Local optimization using Nelder-Mead
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            def callback(xk):
                nonlocal evaluations, best_solution, best_value
                value = func(xk)
                evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = xk
                return evaluations >= self.budget

            options = {'maxiter': remaining_budget, 'xatol': 1e-6, 'fatol': 1e-6}  # Adjusted threshold
            result = minimize(func, best_solution, method='Nelder-Mead', callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution