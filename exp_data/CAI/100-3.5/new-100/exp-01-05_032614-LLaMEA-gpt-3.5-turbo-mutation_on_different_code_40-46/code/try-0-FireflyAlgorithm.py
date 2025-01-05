import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        gamma = lambda r: self.beta0 * np.exp(-self.alpha*r**2)
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        for i in range(self.budget):
            for j in range(self.budget):
                if func(x) < func(x + gamma(np.linalg.norm(x - y)) * (y - x)):
                    x = x + gamma(np.linalg.norm(x - y)) * (y - x)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt