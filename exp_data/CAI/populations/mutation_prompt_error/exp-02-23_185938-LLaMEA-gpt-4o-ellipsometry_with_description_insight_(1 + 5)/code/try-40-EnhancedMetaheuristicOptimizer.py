import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def __call__(self, func):
        initial_sample_count = max(10, self.budget // 8)
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        X = np.array(initial_samples)
        y = np.array([func(sample) for sample in initial_samples])
        self.budget -= len(initial_samples)

        self.gp.fit(X, y)
        best_sample = X[np.argmin(y)]
        best_value = np.min(y)

        while self.budget > 0:
            candidate_sample = self.propose_location(func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_sample)
            self.budget -= 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_sample = candidate_sample

            X = np.vstack((X, candidate_sample))
            y = np.append(y, candidate_value)
            self.gp.fit(X, y)

            if self.budget <= 0:
                return best_sample

        bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        def objective(x):
            return func(x)

        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.8), 'ftol': 1e-7})

        if res.success:
            return res.x
        else:
            return best_sample

    def propose_location(self, lb, ub):
        random_point = np.random.uniform(lb, ub, size=self.dim)
        return random_point