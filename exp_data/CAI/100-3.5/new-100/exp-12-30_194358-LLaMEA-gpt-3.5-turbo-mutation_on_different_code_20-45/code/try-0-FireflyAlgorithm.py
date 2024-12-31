import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y))

    def __call__(self, func):
        # Initialize fireflies randomly
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) + self.attractiveness(fireflies[i], fireflies[j]) * (np.random.rand(self.dim) - 0.5)
            
            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]
                
        return self.f_opt, self.x_opt