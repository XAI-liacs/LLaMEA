import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, sigma_init=0.5):
        self.budget = budget
        self.dim = dim
        self.sigma_init = sigma_init
        self.sigma = np.full(dim, sigma_init)
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        cov = np.eye(self.dim)
        
        for _ in range(self.budget):
            population = np.random.multivariate_normal(mean, cov)
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]
            
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            
            mean = np.mean(population[:int(0.5*len(population))], axis=0)
            cov = np.cov(population[:int(0.5*len(population))].T)
            
        return self.f_opt, self.x_opt