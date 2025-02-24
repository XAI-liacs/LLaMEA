import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class LatinHypercubeBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        evaluations = 0
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        def budgeted_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                raise RuntimeError("Exceeded budget of function evaluations.")
            evaluations += 1
            return func(x)

        num_initial_samples = int(min(12, self.budget // 7) + np.log(self.budget + 2))  # Adjusted portion of the budget
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(num_initial_samples)
        initial_guesses = qmc.scale(sample, lb, ub)

        best_result = None

        for initial_guess in initial_guesses:
            try:
                result = minimize(budgeted_func, initial_guess, method='BFGS', options={'maxiter': self.budget // 4})
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            except RuntimeError:
                break

        if best_result:
            final_result = minimize(budgeted_func, best_result.x, method='Nelder-Mead', options={'maxiter': self.budget - evaluations})
            if final_result.fun < best_result.fun:
                best_result = final_result

        return best_result.x if best_result else None