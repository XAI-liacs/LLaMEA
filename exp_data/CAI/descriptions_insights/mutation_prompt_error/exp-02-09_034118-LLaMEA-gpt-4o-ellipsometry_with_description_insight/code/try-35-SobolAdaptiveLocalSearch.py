import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class SobolAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial sampling using Sobol sequence for better space coverage
        num_initial_samples = min(15, self.budget // 2)  # Increased initial samples for better diversity
        sobol_sampler = Sobol(d=self.dim, scramble=True)  # Scrambled Sobol to introduce variability
        initial_samples = sobol_sampler.random_base2(m=int(np.log2(num_initial_samples)))
        initial_samples = bounds[0] + (bounds[1] - bounds[0]) * initial_samples

        for x0 in initial_samples:
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
                if evaluations >= self.budget:
                    return True
                value = func(xk)
                evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = xk
                current_tol = (1e-8) * (remaining_budget / self.budget)
                return np.abs(value - best_value) < current_tol or evaluations % 10 == 0  # Adaptive restart

            options = {'maxiter': remaining_budget, 'xatol': 1e-8 * (remaining_budget/self.budget), 'fatol': 1e-8}
            result = minimize(func, best_solution, method='Nelder-Mead', callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution