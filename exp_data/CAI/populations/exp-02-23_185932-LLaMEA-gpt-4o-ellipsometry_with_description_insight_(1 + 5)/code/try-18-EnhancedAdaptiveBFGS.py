import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedAdaptiveBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = float('inf')

        # Sobol sequence for initial guesses
        num_initial_guesses = min(5, self.budget // self.dim)
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        initial_guesses = sobol_sampler.random_base2(m=int(np.log2(num_initial_guesses))) * (bounds[1] - bounds[0]) + bounds[0]

        for guess in initial_guesses:
            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': self.budget - self.evaluations})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Break early if budget exceeded
            if self.evaluations >= self.budget:
                break

            # Adaptive boundary adjustment with dynamic constraint relaxation
            bounds_range = 0.1 * (bounds[1] - bounds[0])
            bounds[0] = np.maximum(func.bounds.lb, best_solution - bounds_range)
            bounds[1] = np.minimum(func.bounds.ub, best_solution + bounds_range)

        return best_solution

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            # Avoid further evaluations and terminate early
            return float('inf')