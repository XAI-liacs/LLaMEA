import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 1.0
        self.beta = 2.0
        self.gamma = 0.2
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
    
    def attractiveness(self, x, y):
        return np.exp(-self.beta * np.linalg.norm(x - y))
    
    def move_firefly(self, i):
        for j in range(self.population_size):
            if func(self.population[j]) < func(self.population[i]):
                r = np.linalg.norm(self.population[i] - self.population[j])
                beta0 = self.alpha * np.exp(-self.gamma * r**2)
                self.population[i] = self.population[i] + beta0 * (self.population[j] - self.population[i]) + np.random.uniform(-0.1, 0.1, self.dim)
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                self.move_firefly(i)
        best_solution = min(self.population, key=lambda x: func(x))
        return best_solution