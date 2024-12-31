import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5  # Alpha parameter controls randomness
        self.beta0 = 1.0  # Initial attractiveness
        self.gamma = 0.01  # Step size scaling factor
        self.pop_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.fitness = np.array([np.Inf] * self.pop_size)
        self.best_fitness = np.Inf
        self.best_solution = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_firefly(self, i, j, func):
        r = np.linalg.norm(self.population[i] - self.population[j])  # Distance between fireflies
        beta = self.attractiveness(r)
        epsilon = self.alpha * (np.random.rand(self.dim) - 0.5)  # Random step
        self.population[i] += beta * (self.population[j] - self.population[i]) + epsilon
        self.population[i] = np.clip(self.population[i], -5.0, 5.0)
        new_fitness = func(self.population[i])
        if new_fitness < self.fitness[i]:
            self.fitness[i] = new_fitness

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if self.fitness[j] < self.fitness[i]:  # Firefly i is less bright than j
                        self.move_firefly(i, j, func)
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

        return self.best_fitness, self.best_solution