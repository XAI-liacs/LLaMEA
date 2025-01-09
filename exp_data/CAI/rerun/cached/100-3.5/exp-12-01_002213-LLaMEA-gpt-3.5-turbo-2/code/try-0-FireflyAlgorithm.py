import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        alpha = 0.2
        beta0 = 1.0
        gamma = 1.0
        population_size = 20
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            for i in range(population_size):
                for j in range(population_size):
                    if fitness[j] < fitness[i]:
                        attractiveness = beta0 * np.exp(-gamma * np.linalg.norm(population[j] - population[i])**2)
                        population[i] += alpha * attractiveness * (population[j] - population[i]) + 0.01 * np.random.normal(0, 1, self.dim)
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]