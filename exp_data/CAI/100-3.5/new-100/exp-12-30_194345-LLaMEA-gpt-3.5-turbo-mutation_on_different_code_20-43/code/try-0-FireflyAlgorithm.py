import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.full(budget, np.Inf)
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, light_intensity):
        return self.alpha * np.exp(-self.beta * light_intensity)

    def move_fireflies(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[j]) < func(self.population[i]):
                    attractiveness_i = self.attractiveness(func(self.population[i]))
                    attractiveness_j = self.attractiveness(func(self.population[j]))

                    self.population[i] += attractiveness_i * (self.population[j] - self.population[i])

    def __call__(self, func):
        for i in range(self.budget):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]

            self.move_fireflies(func)

        return self.f_opt, self.x_opt