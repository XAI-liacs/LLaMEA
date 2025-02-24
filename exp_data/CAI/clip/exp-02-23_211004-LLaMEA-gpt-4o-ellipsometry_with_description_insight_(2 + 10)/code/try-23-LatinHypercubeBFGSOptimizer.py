import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class LatinHypercubeBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        evaluations = 0  # Keep track of the number of function evaluations

        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        def budgeted_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                raise RuntimeError("Exceeded budget of function evaluations.")
            evaluations += 1
            return func(x)

        num_initial_samples = min(10, self.budget // 10)  # Use a portion of the budget
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(num_initial_samples)
        initial_guesses = qmc.scale(sample, lb, ub)

        best_result = None

        for initial_guess in initial_guesses:
            try:
                result = minimize(budgeted_func, initial_guess, method='BFGS', 
                                  options={'maxiter': self.budget // 4, 'gtol': 1e-7})
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            except RuntimeError:
                break

        # Adaptive boundary adjustment for further refinement
        if best_result:
            new_lb = np.maximum(lb, best_result.x - 0.1 * (ub - lb))
            new_ub = np.minimum(ub, best_result.x + 0.1 * (ub - lb))
            result = minimize(budgeted_func, best_result.x, method='BFGS', 
                              bounds=list(zip(new_lb, new_ub)), options={'maxiter': self.budget // 2})
            if result.fun < best_result.fun:
                best_result = result

        return best_result.x if best_result else None