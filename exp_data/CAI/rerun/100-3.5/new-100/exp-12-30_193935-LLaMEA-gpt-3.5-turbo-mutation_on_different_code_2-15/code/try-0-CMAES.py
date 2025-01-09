import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, init_mean=None, init_sigma=1.0):
        self.budget = budget
        self.dim = dim
        self.init_mean = init_mean if init_mean is not None else np.zeros(dim)
        self.init_sigma = init_sigma
        self.sigma = init_sigma
        self.mean = self.init_mean
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        mu = int(4 + np.floor(3 * np.log(self.dim)))
        cov = np.eye(self.dim)
        samples = np.random.multivariate_normal(self.mean, self.sigma**2 * cov, mu)
        
        for t in range(self.budget):
            fitness = np.array([func(x) for x in samples])
            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = samples[np.argmin(fitness)]
            
            sorted_indices = np.argsort(fitness)
            best_samples = samples[sorted_indices[:mu]]
            self.mean = np.mean(best_samples, axis=0)
            cov = np.cov(best_samples, rowvar=False)
            self.sigma *= np.exp(1.0 / self.dim * (np.linalg.norm(self.mean - best_samples.mean(axis=0)) / 0.3 - 1))
            samples = np.random.multivariate_normal(self.mean, self.sigma**2 * cov, mu)
        
        return self.f_opt, self.x_opt