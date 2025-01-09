import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta_min=0.2, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta_min * np.exp(-self.gamma * r**2)
    
    def move_firefly(self, x_i, x_j, attractiveness):
        return x_i + attractiveness * (x_j - x_i) + self.alpha * (np.random.rand(self.dim) - 0.5)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = self.attractiveness(r)
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta)
            
            for k in range(self.budget):
                if func(fireflies[k]) < self.f_opt:
                    self.f_opt = func(fireflies[k])
                    self.x_opt = fireflies[k]
        
        return self.f_opt, self.x_opt