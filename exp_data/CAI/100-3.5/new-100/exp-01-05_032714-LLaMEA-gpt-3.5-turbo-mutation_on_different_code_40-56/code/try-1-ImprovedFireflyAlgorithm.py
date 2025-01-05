import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha_min=0.2, alpha_max=0.9, beta_min=0.2, beta_max=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.array([np.Inf] * budget)

    def attractiveness(self, r, beta):
        return beta * np.exp(-r)

    def move_firefly(self, source, target, alpha, beta):
        r = np.linalg.norm(source - target)
        new_beta = self.attractiveness(r, beta)
        return target + alpha * new_beta * (source - target) + 0.01 * np.random.randn(self.dim)

    def __call__(self, func):
        for i in range(self.budget):
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * i / self.budget
            beta = self.beta_min + (self.beta_max - self.beta_min) * i / self.budget
            for j in range(self.budget):
                if func(self.population[i]) < func(self.population[j]):
                    self.population[i] = self.move_firefly(self.population[i], self.population[j], alpha, beta)

        idx = np.argmin([func(x) for x in self.population])
        return func(self.population[idx]), self.population[idx]