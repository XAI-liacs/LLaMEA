import numpy as np
import math

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * math.exp(-self.gamma * r**2)

    def move_firefly(self, x_i, x_j):
        r = np.linalg.norm(x_i - x_j)
        beta = self.attractiveness(r)
        return x_i + beta * (x_j - x_i) + self.alpha * (np.random.rand(self.dim) - 0.5)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(i+1, self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])
        
        best_idx = np.argmin([func(firefly) for firefly in fireflies])
        self.f_opt = func(fireflies[best_idx])
        self.x_opt = fireflies[best_idx]
        
        return self.f_opt, self.x_opt