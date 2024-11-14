import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = None

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        def attractiveness(i, j):
            r = np.linalg.norm(self.population[i] - self.population[j])
            return 1 / (1 + r)

        def move_fireflies():
            new_population = np.copy(self.population)
            for i in range(self.budget):
                alpha = 1.0 / (1 + i)
                gamma = 0.5 / (1 + i)
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        new_population[i] += alpha * np.exp(-1 * np.linalg.norm(self.population[j] - self.population[i]) ** 2) * (self.population[j] - self.population[i]) + gamma * np.random.uniform(-1, 1, self.dim)
            self.population = new_population

        self.population = initialize_population()
        for _ in range(self.budget):
            move_fireflies()
        best_solution = min(self.population, key=lambda x: func(x))
        return best_solution