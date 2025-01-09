import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def attractiveness(self, r):
        return 1 / (1 + r)

    def move_firefly(self, current, brightest, beta0, gamma):
        r = np.linalg.norm(brightest - current)
        beta = beta0 * np.exp(-gamma * r**2)
        step = beta * (np.random.rand(self.dim) - 0.5)
        return current + step

    def __call__(self, func):
        population = 5 + int(10 * np.sqrt(self.dim))
        beta0 = 1.0
        gamma = 0.2
        population_positions = np.random.uniform(-5.0, 5.0, (population, self.dim))
        population_fitness = np.array([func(x) for x in population_positions])
        for _ in range(self.budget):
            for i in range(population):
                for j in range(population):
                    if population_fitness[j] < population_fitness[i]:
                        population_positions[i] = self.move_firefly(population_positions[i], population_positions[j], beta0, gamma)
                        population_fitness[i] = func(population_positions[i])
        best_idx = np.argmin(population_fitness)
        return population_positions[best_idx]