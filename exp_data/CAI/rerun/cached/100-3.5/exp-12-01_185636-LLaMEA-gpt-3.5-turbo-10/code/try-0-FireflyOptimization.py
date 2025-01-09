import numpy as np

class FireflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        alpha = 0.2  # Attraction coefficient
        beta_min = 0.2  # Minimum value for beta
        beta_max = 1.0  # Maximum value for beta
        gamma = 1.0  # Light absorption coefficient
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if fitness[i] > fitness[j]:
                        beta = beta_min + (beta_max - beta_min) * np.exp(-gamma * np.linalg.norm(population[i] - population[j])**2)
                        population[i] += alpha * (population[j] - population[i]) + beta * np.random.uniform(-5.0, 5.0, self.dim)
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
        
        best_index = np.argmin(fitness)
        return population[best_index]