import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            
            for j in range(self.budget):
                y = np.random.uniform(func.bounds.lb, func.bounds.ub)
                fy = func(y)
                
                if fy < f:
                    x = y
                    f = fy
            
            for j in range(self.budget):
                for k in range(self.budget):
                    r = np.linalg.norm(x - y)
                    beta = self.attractiveness(r)
                    x = x + self.alpha * (np.random.uniform(func.bounds.lb, func.bounds.ub) - x) + beta * (y - x)
                    f = func(x)
                    
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt