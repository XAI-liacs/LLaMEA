import numpy as np
from scipy.optimize import minimize, basinhopping

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.init_sample_ratio = 0.2  # Added for dynamic sampling adjustment
        self.local_search_method = 'L-BFGS-B'  # Changed method for improved convergence

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        init_sample_size = int(self.init_sample_ratio * self.budget)
        init_points = np.random.uniform(lb, ub, size=(init_sample_size, self.dim))  # Adjusted sampling size

        best_solution = None
        best_value = float('inf')

        for point in init_points:
            if self.evaluations >= self.budget:
                break

            result = minimize(self.bounded_func(func, lb, ub), point, method=self.local_search_method,
                              options={'maxfun': self.budget - self.evaluations})

            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))  # Adjusted bounds for greater flexibility
            ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))

        # Implemented hybrid optimization with basinhopping
        minimizer_kwargs = {"method": self.local_search_method, 
                            "bounds": np.array(list(zip(lb, ub)))}
        result = basinhopping(self.bounded_func(func, lb, ub), best_solution, 
                              minimizer_kwargs=minimizer_kwargs, niter=10)

        self.evaluations += result.nfev
        if result.fun < best_value:
            best_solution = result.x

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds