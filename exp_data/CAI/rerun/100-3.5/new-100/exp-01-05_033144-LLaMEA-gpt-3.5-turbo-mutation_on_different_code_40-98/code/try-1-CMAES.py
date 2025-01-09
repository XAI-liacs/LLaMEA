import numpy as np
from scipy.stats import truncnorm

class CMAES:
    def __init__(self, budget=10000, dim=10, sigma_init=1.0, sigma_decay=0.995):
        self.budget = budget
        self.dim = dim
        self.sigma = sigma_init
        self.sigma_decay = sigma_decay
        self.mean = np.random.uniform(-5.0, 5.0, dim)
        self.cov_matrix = np.eye(dim)
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for _ in range(self.budget):
            population = truncnorm.rvs(-2, 2, loc=0, scale=self.sigma, size=(10*self.dim, self.dim)) @ self.cov_matrix.T + self.mean
            fitness = [func(individual) for individual in population]
            idx_best = np.argmin(fitness)
            if fitness[idx_best] < self.f_opt:
                self.f_opt = fitness[idx_best]
                self.x_opt = population[idx_best]
                
            # Update mean and covariance matrix
            weights = np.exp(-(np.arange(10*self.dim)+1) / (10*self.dim/2))
            weights /= np.sum(weights)
            weighted_population = population * weights[:, np.newaxis]
            self.mean = np.sum(weighted_population, axis=0)
            demeaned_population = population - self.mean
            self.cov_matrix = np.dot(weights * demeaned_population.T, demeaned_population)
            
            # Adaptive step size control
            self.sigma *= self.sigma_decay
            
        return self.f_opt, self.x_opt