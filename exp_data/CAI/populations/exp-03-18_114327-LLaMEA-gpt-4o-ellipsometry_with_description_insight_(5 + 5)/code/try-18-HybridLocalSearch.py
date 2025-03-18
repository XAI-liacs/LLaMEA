import numpy as np
from scipy.optimize import minimize

class HybridLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]

        num_initial_samples = min(max(5, self.budget // 15), self.budget)  # Increased sampling frequency
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

            # Utilize random neighborhood exploration
            neighborhood_radius = 0.1 * (upper_bound - lower_bound)
            neighbors = np.random.uniform(best_sample - neighborhood_radius, best_sample + neighborhood_radius, (5, self.dim))
            for neighbor in neighbors:
                value = func(neighbor)
                self.budget -= 1
                if value < best_value:
                    best_value = value
                    best_sample = neighbor

            if self.budget < 20:
                radius = 0.03 * (upper_bound - lower_bound)
            else:
                radius = 0.07 * (upper_bound - lower_bound)
            bounds = [(max(lower_bound[i], best_sample[i] - radius[i]), min(upper_bound[i], best_sample[i] + radius[i])) for i in range(self.dim)]

        return best_sample, best_value