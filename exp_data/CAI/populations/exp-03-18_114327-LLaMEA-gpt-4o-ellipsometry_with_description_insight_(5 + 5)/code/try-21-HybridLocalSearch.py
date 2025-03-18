import numpy as np
from scipy.optimize import minimize

class HybridLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.initial_budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]

        num_initial_samples = max(5, int(0.05 * self.initial_budget))  # Adaptive number of initial samples
        samples = np.random.uniform(lower_bound, upper_bound, (num_initial_samples, self.dim))
        evaluations = [func(sample) for sample in samples]
        self.budget -= num_initial_samples

        best_idx = np.argmin(evaluations)
        best_sample = samples[best_idx]
        best_value = evaluations[best_idx]

        while self.budget > 0:
            def local_func(x):
                return func(x)

            result = minimize(local_func, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget})
            self.budget -= result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

            radius = 0.05 * (upper_bound - lower_bound)  # 5% of the original domain size for refined search
            bounds = [(max(lower_bound[i], best_sample[i] - radius[i]), min(upper_bound[i], best_sample[i] + radius[i])) for i in range(self.dim)]

        return best_sample, best_value