import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class SobolAdaptiveHybridSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial sampling using Sobol sequence for better space coverage
        num_initial_samples = min(10, self.budget // 2)
        sobol_sampler = Sobol(d=self.dim)
        initial_samples = sobol_sampler.random_base2(m=int(np.log2(num_initial_samples)))
        initial_samples = bounds[0] + (bounds[1] - bounds[0]) * initial_samples

        for x0 in initial_samples:
            value = func(x0)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

        # Hybrid optimization using Nelder-Mead and BFGS
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            def callback(xk):
                nonlocal evaluations, best_solution, best_value
                if evaluations >= self.budget:
                    return True
                value = func(xk)
                evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = xk

            options_nm = {'maxiter': remaining_budget // 2, 'xatol': 1e-8, 'fatol': 1e-8}
            result_nm = minimize(func, best_solution, method='Nelder-Mead', callback=callback, options=options_nm, bounds=bounds.T)
            
            if result_nm.fun < best_value:
                best_solution = result_nm.x

            options_bfgs = {'maxiter': remaining_budget // 2}
            result_bfgs = minimize(func, best_solution, method='BFGS', options=options_bfgs, bounds=bounds.T)

            if result_bfgs.fun < best_value:
                best_solution = result_bfgs.x

        return best_solution