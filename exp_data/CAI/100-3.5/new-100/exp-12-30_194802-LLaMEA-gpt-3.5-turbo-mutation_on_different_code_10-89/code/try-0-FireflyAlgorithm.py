import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.pop_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.fitness = np.array([np.Inf] * self.pop_size)
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_firefly(self, i, func):
        for j in range(self.pop_size):
            if self.fitness[i] > self.fitness[j]:
                r = np.linalg.norm(self.population[i] - self.population[j])
                beta = self.attractiveness(r)
                self.population[i] += beta * (self.population[j] - self.population[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                self.fitness[i] = func(self.population[i])
                
    def __call__(self, func):
        for i in range(self.budget):
            for k in range(self.pop_size):
                self.move_firefly(k, func)
                
                for f_idx, f_value in enumerate(self.fitness):
                    if f_value < self.f_opt:
                        self.f_opt = f_value
                        self.x_opt = self.population[f_idx]

        return self.f_opt, self.x_opt