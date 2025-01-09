import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.alpha = 0.2
        self.beta = 1.0
        self.gamma = 0.05
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
    
    def attractiveness(self, light_intensity):
        return self.beta * np.exp(-self.gamma * light_intensity)
    
    def move_fireflies(self, func):
        evaluated = 0
        while evaluated < self.budget:
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(self.population[i]) < func(self.population[j]):
                        self.population[i] += self.attractiveness(func(self.population[j])) * (self.population[j] - self.population[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                evaluated += 1
        return self.population[np.argmin([func(ind) for ind in self.population])]

    def __call__(self, func):
        return self.move_fireflies(func)