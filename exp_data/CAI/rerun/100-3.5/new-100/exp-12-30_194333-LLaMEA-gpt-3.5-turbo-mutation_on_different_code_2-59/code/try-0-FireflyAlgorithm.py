import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def attractiveness(self, light_intensity):
        return self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * light_intensity)
    
    def move_fireflies(self, current_func):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if current_func(self.population[i]) < current_func(self.population[j]):
                    attractiveness_ij = self.attractiveness(current_func(self.population[i]))
                    self.population[j] += attractiveness_ij * (self.population[i] - self.population[j]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                    
    def __call__(self, func):
        for _ in range(self.budget):
            self.move_fireflies(func)
        
        best_index = np.argmin([func(x) for x in self.population])
        return func(self.population[best_index]), self.population[best_index]