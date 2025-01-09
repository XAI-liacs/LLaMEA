import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.2  # Alpha parameter
        self.beta0 = 1.0  # Initial beta parameter
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return np.exp(-np.linalg.norm(x - y)**2)

    def move_firefly(self, x, best_x, i):
        beta = self.beta0 * np.exp(-self.alpha * (i+1) / self.budget)
        new_x = x + beta * (best_x - x) + 0.01 * np.random.normal(size=self.dim)
        return np.clip(new_x, -5.0, 5.0)

    def update_parameters(self, i):
        self.alpha = self.alpha * 0.95
        self.beta0 = self.beta0 * 0.99

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], i)
                    
            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]
            
            self.update_parameters(i)
            
        return self.f_opt, self.x_opt