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
        return self.beta0 * np.exp(-self.gamma * r)

    def __call__(self, func):
        # Initialize fireflies randomly
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(x[j]) < func(x[i]):
                        r = np.linalg.norm(x[i] - x[j])
                        beta = self.attractiveness(r)
                        x[i] += self.alpha * (x[j] - x[i]) * beta
                        
            # Update global best
            for i in range(self.budget):
                f = func(x[i])
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x[i]
            
        return self.f_opt, self.x_opt