import numpy as np
from scipy.optimize import minimize

class DifferentiallyInformedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Differential sampling for informed initial guesses
        num_initial_samples = min(15, self.budget // 3)  # Increased initial samples with differential strategy
        samples = np.random.uniform(bounds[0], bounds[1], (num_initial_samples, self.dim))
        diffs = np.diff(samples, axis=0)
        initial_guesses = samples[:-1] + 0.5 * diffs

        for x0 in initial_guesses:
            value = func(x0)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

        # Local optimization using modified BFGS with dynamic step
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
                return False

            options = {'maxiter': remaining_budget, 'gtol': 1e-6}
            result = minimize(func, best_solution, method='BFGS', callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution