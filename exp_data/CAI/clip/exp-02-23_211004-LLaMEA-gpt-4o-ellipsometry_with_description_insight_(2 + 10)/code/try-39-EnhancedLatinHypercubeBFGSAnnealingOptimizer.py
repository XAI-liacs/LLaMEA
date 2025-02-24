import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedLatinHypercubeBFGSAnnealingOptimizer:
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

        num_initial_samples = min(10, self.budget // 10)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(num_initial_samples)
        initial_guesses = qmc.scale(sample, lb, ub)

        best_result = None

        for initial_guess in initial_guesses:
            try:
                result = minimize(budgeted_func, initial_guess, method='BFGS', options={'maxiter': self.budget // 4})
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
                    # Adaptive bounds tightening
                    lb = np.maximum(lb, best_result.x - 0.1 * (ub - lb))
                    ub = np.minimum(ub, best_result.x + 0.1 * (ub - lb))
                    
                # Simulated annealing step for potential global exploration
                for _ in range(5):  # Define number of annealing iterations
                    temp_initial_guess = np.random.uniform(lb, ub)
                    temp_result = minimize(budgeted_func, temp_initial_guess, method='BFGS', options={'maxiter': self.budget // 20})
                    if temp_result.fun < best_result.fun:
                        best_result = temp_result
                        lb = np.maximum(lb, best_result.x - 0.1 * (ub - lb))
                        ub = np.minimum(ub, best_result.x + 0.1 * (ub - lb))
                    
            except RuntimeError:
                break

        return best_result.x if best_result else None