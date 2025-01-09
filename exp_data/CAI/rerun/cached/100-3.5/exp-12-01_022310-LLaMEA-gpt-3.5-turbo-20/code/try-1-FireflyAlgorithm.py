import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.8
        self.beta_min = 0.2
        self.gamma = 1.0
        self.eta = 0.1  # Parameter for Levy flight
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def attractiveness(self, x_i, x_j):
        return np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def levy_flight(self):
        return np.random.standard_cauchy(self.dim) / np.power(np.abs(np.random.normal(0, 1, self.dim)), 1/self.eta)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[j]) < func(population[i]):
                        population[i] += self.alpha * self.attractiveness(population[i], population[j]) * (population[j] - population[i]) + self.beta_min * self.levy_flight()
            population = np.clip(population, self.lower_bound, self.upper_bound)
        return population[np.argmin([func(x) for x in population])]