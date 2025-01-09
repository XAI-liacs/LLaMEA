import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, i, j):
        return self.beta0 * np.exp(-self.alpha * np.linalg.norm(self.population[i] - self.population[j]))

    def move_firefly(self, i, j):
        r = np.linalg.norm(self.population[i] - self.population[j])
        beta = self.beta0 * np.exp(-self.alpha * r**2)
        epsilon = np.random.uniform(-1, 1, self.dim)
        self.population[i] += beta * (self.population[j] - self.population[i]) + 0.01 * epsilon

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i, j)
        return min(self.population, key=func)