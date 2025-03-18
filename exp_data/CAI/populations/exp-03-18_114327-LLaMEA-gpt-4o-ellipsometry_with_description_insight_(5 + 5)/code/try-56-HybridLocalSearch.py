import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Unpack bounds
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]

        # Adaptive initial sampling
        num_initial_samples = min(max(10, self.budget // 15), self.budget)  # Adjusted sampling frequency
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(num_initial_samples)))
        samples = lower_bound + samples * (upper_bound - lower_bound)
        evaluations = [func(sample) for sample in samples]
        self.budget -= num_initial_samples

        # Take the best initial sample as a starting point for local optimization
        best_idx = np.argmin(evaluations)
        best_sample = samples[best_idx]
        best_value = evaluations[best_idx]

        # Iteratively refine using BFGS
        while self.budget > 0:
            # Define the local optimization function
            def local_func(x):
                return func(x)

            # Optimize using BFGS starting from the best known sample
            result = minimize(local_func, best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget})

            # Update the budget with the number of function evaluations used
            self.budget -= result.nfev

            # Check and update the best found solution
            if result.fun < best_value:
                best_value = result.fun
                best_sample = result.x

            # Dynamically adjust bounds around the current best solution for finer search
            radius = 0.05 * (upper_bound - lower_bound) if self.budget < 20 else 0.1 * (upper_bound - lower_bound)
            bounds = [(max(lower_bound[i], best_sample[i] - radius[i]), min(upper_bound[i], best_sample[i] + radius[i])) for i in range(self.dim)]

        return best_sample, best_value