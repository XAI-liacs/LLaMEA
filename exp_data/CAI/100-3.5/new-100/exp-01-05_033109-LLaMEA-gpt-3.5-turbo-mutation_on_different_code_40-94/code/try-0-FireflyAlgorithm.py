import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.01):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        self.fitness = np.array([np.Inf] * self.budget)
        self.best_fitness_idx = np.argmin(self.fitness)

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if self.fitness[j] < self.fitness[i]:
                        r = np.linalg.norm(self.population[i] - self.population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        epsilon = self.alpha * (np.random.rand(self.dim) - 0.5)
                        self.population[i] += beta * (self.population[j] - self.population[i]) + epsilon
                        self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                        self.fitness[i] = func(self.population[i])
                        if self.fitness[i] < self.fitness[self.best_fitness_idx]:
                            self.best_fitness_idx = i

        return self.fitness[self.best_fitness_idx], self.population[self.best_fitness_idx]