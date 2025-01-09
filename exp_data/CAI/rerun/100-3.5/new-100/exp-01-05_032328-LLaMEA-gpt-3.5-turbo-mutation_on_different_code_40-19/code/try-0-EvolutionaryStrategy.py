import numpy as np

class EvolutionaryStrategy:
    def __init__(self, budget=10000, dim=10, mu=10, lambda_=30, sigma_init=0.1, sigma_adapt=0.1):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma_init
        self.sigma_adapt = sigma_adapt
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def mutate(x, sigma):
            return x + np.random.normal(0, sigma, size=len(x))

        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.mu, self.dim))
        for i in range(self.budget // self.lambda_):
            offspring = np.array([mutate(x[i % self.mu], self.sigma) for i in range(self.lambda_)])
            f_vals = np.array([func(off) for off in offspring])
            best_idx = np.argmin(f_vals)
            if f_vals[best_idx] < self.f_opt:
                self.f_opt = f_vals[best_idx]
                self.x_opt = offspring[best_idx]
            selected = offspring[np.argsort(f_vals)[:self.mu]]
            x_mean = np.mean(selected, axis=0)
            self.sigma *= np.exp(self.sigma_adapt * np.random.normal(0, 1))
            x = np.tile(x_mean, (self.lambda_ // self.mu, 1))
            x += self.sigma * np.random.normal(0, 1, x.shape)

        return self.f_opt, self.x_opt