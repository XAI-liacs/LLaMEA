import numpy as np

class EvolutionStrategy:
    def __init__(self, budget=10000, dim=10, lambda_=10, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.lambda_ = lambda_
        self.sigma = sigma
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.lambda_, self.dim))
        for _ in range(self.budget // self.lambda_):
            offspring = x + np.random.normal(0, self.sigma, size=x.shape)
            f = np.array([func(ind) for ind in offspring])
            best_idx = np.argmin(f)
            if f[best_idx] < self.f_opt:
                self.f_opt = f[best_idx]
                self.x_opt = offspring[best_idx]
            x = offspring[best_idx]

        return self.f_opt, self.x_opt