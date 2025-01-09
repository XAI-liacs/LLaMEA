import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=0.2, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.fitness = np.inf * np.ones(budget)

    def attractiveness(self, r):
        return self.alpha * np.exp(-self.beta * r**2)

    def move_fireflies(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if self.fitness[i] > func(self.population[j]):
                    r = np.linalg.norm(self.population[i] - self.population[j])
                    beta = self.attractiveness(r)
                    self.population[i] += beta * (self.population[j] - self.population[i])
                    self.fitness[i] = func(self.population[i])
            self.alpha *= 0.95  # Adapt alpha dynamically

    def __call__(self, func):
        for _ in range(self.budget):
            self.move_fireflies(func)
        
        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], self.population[best_idx]