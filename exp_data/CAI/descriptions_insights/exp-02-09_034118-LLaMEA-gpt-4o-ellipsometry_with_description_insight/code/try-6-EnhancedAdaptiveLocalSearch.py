import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial exploration using Sobol sequence
        num_initial_samples = min(5, self.budget // 2)
        sobol_engine = Sobol(d=self.dim, scramble=True)
        for _ in range(num_initial_samples):
            x0 = sobol_engine.random_base2(m=int(np.log2(num_initial_samples)))
            x0 = func.bounds.lb + x0 * (func.bounds.ub - func.bounds.lb)
            value = func(x0[0])
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0[0]

        # Local optimization using BFGS
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

            options = {'maxiter': remaining_budget, 'gtol': 1e-8}
            result = minimize(func, best_solution, method='BFGS', callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution