import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0, gamma=0.01):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.array([np.Inf] * budget)

    def attractiveness(self, light_intensity):
        return self.alpha * np.exp(-self.beta * light_intensity)

    def move_firefly(self, current, target):
        distance = np.linalg.norm(current - target)
        step_size = self.gamma * distance
        new_position = current + step_size * (target - current) / distance
        return np.clip(new_position, -5.0, 5.0)

    def __call__(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[i]) < func(self.population[j]):
                    attractiveness_i = self.attractiveness(func(self.population[i]))
                    self.population[i] = self.move_firefly(self.population[i], self.population[j])
                    self.fitness[i] = func(self.population[i])

        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], self.population[best_idx]