import numpy as np

class GreyWolfOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 5
        self.alpha = np.zeros(dim)
        self.beta = np.zeros(dim)
        self.delta = np.zeros(dim)
        self.pop = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for _ in range(self.budget):
            a = 2 - 2 * _ / self.budget  # Formula for linearly decreasing a
            
            for i in range(self.population_size):
                fitness = func(self.pop[i])
                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = self.pop[i]
                
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha[j] - self.pop[i, j])
                    X1 = self.alpha[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta[j] - self.pop[i, j])
                    X2 = self.beta[j] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta[j] - self.pop[i, j])
                    X3 = self.delta[j] - A3 * D_delta
                    
                    self.pop[i, j] = (X1 + X2 + X3) / 3
        
        return self.f_opt, self.x_opt