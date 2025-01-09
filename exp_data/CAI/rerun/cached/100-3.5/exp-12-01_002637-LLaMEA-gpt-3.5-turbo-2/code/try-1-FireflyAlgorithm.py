import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5
        self.beta_min = 0.2
        self.gamma = 1.0

    def attractiveness(self, light_intensity):
        return self.beta_min * np.exp(-self.gamma * light_intensity)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        light_intensity = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if light_intensity[j] < light_intensity[i]:
                        attractiveness_ij = self.attractiveness(light_intensity[i])
                        population[i] += attractiveness_ij * (population[j] - population[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        light_intensity[i] = func(population[i])
                        
        best_index = np.argmin(light_intensity)
        best_solution = population[best_index]
        return best_solution