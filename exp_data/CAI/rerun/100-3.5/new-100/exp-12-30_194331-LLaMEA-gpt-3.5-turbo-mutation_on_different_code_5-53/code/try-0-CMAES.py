import numpy as np
from scipy.stats import multivariate_normal

class CMAES:
    def __init__(self, budget=10000, dim=10, sigma=0.3):
        self.budget = budget
        self.dim = dim
        self.sigma = sigma
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        cov = np.eye(self.dim)
        
        for i in range(self.budget):
            samples = np.random.multivariate_normal(mean, cov, 5)
            fitness = np.array([func(x) for x in samples])
            best_idx = np.argmin(fitness)
            
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = samples[best_idx]
            
            # Update mean and covariance using CMA-ES
            mean = mean + self.sigma * np.dot(cov, np.mean(samples, axis=0) - mean)
            cov = cov + self.sigma * np.dot((samples - mean).T, samples - mean) / self.dim
            
        return self.f_opt, self.x_opt