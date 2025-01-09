import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return np.exp(-self.alpha * np.linalg.norm(x - y))

    def __call__(self, func):
        n_fireflies = self.budget
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(n_fireflies, self.dim))
        
        for i in range(self.budget):
            for j in range(n_fireflies):
                for k in range(n_fireflies):
                    if func(fireflies[j]) < func(fireflies[k]):
                        beta = self.beta0 * np.exp(-self.alpha * np.linalg.norm(fireflies[j] - fireflies[k])**2)
                        fireflies[j] += beta * (fireflies[k] - fireflies[j]) + np.random.normal(0, 1, self.dim)
            
            for j in range(n_fireflies):
                f = func(fireflies[j])
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = fireflies[j]
                    
        return self.f_opt, self.x_opt