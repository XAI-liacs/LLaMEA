import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.pop_size = 20
        self.pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
    
    def attractiveness(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)
    
    def move_firefly(self, x, y, beta):
        r = np.random.uniform(-1, 1, size=self.dim)
        return x + beta * (y - x) + self.alpha * r
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(self.pop[j]) < func(self.pop[i]):
                        beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(self.pop[i] - self.pop[j])**2)
                        self.pop[i] = self.move_firefly(self.pop[i], self.pop[j], beta)
        
        best_idx = np.argmin([func(x) for x in self.pop])
        return func(self.pop[best_idx]), self.pop[best_idx]