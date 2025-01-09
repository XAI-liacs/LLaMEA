import numpy as np
from scipy.stats import truncnorm

class CMA_ES:
    def __init__(self, budget=10000, dim=10, sigma_init=1.0, learning_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.sigma = sigma_init
        self.mean = np.zeros(dim)
        self.cov_matrix = np.eye(dim)
        self.f_opt = np.Inf
        self.x_opt = None
        self.learning_rate = learning_rate

    def __call__(self, func):
        for i in range(self.budget):
            samples = truncnorm.rvs(-2, 2, loc=0, scale=self.sigma, size=(self.dim,))
            population = np.outer(samples, np.linalg.cholesky(self.cov_matrix))
            evals = [func(self.mean + x) for x in population]
            best_idx = np.argmin(evals)
            
            if evals[best_idx] < self.f_opt:
                self.f_opt = evals[best_idx]
                self.x_opt = self.mean + population[best_idx]
                
            sorted_idxs = np.argsort(evals)
            population = population[sorted_idxs]
            improvements = population[:int(self.learning_rate*len(population))] - self.mean
            self.mean += np.mean(improvements, axis=0)
            self.cov_matrix = np.cov(population.T)

        return self.f_opt, self.x_opt