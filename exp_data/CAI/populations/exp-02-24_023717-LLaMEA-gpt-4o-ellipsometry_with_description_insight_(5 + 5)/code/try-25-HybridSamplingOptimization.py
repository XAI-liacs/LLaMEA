import numpy as np
from scipy.optimize import differential_evolution

class HybridSamplingOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Latin Hypercube Sampling for initial diverse guesses
        num_initial_samples = min(10, self.budget // 5)
        samples = self._latin_hypercube_sampling(lb, ub, num_initial_samples)

        for sample in samples:
            if evaluations >= self.budget:
                break
            result = self._optimize_with_de(func, sample, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution

    def _latin_hypercube_sampling(self, lb, ub, num_samples):
        samples = np.empty((num_samples, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(num_samples)
            samples[:, i] = lb[i] + (ub[i] - lb[i]) * (perm + np.random.rand(num_samples)) / num_samples
        return samples

    def _optimize_with_de(self, func, start_point, lb, ub):
        result = differential_evolution(
            func,
            bounds=list(zip(lb, ub)),
            strategy='best1bin',
            maxiter=1,  # Run each DE iteration with a single generation
            popsize=15,
            init=[start_point],
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            polish=False
        )
        return result