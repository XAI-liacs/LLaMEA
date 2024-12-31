import numpy as np

class FireworksAlgorithmImproved:
    def __init__(self, budget=10000, dim=10, sparks=5, alpha=0.04, beta=3, adapt_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.sparks = sparks
        self.alpha = alpha
        self.beta = beta
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.sparks, self.dim))
        for i in range(self.budget):
            for j in range(self.sparks):
                x = population[j]
                a = np.random.uniform(-self.alpha, self.alpha, self.dim)
                b = np.random.uniform(-self.beta, self.beta, self.dim)
                x_new = np.clip(x + a * (x - population[np.random.randint(self.sparks)]) + b, func.bounds.lb, func.bounds.ub)
                f = func(x_new)
                if f < func(x):
                    population[j] = x_new
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x_new
            self.alpha *= self.adapt_rate
            self.beta *= self.adapt_rate
                    
        return self.f_opt, self.x_opt