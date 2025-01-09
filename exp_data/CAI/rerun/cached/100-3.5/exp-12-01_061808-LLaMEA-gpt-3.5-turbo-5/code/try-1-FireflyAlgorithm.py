import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.2
        self.beta = 1.0
        self.gamma = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
    
    def attractiveness(self, light_intensity):
        return self.beta * np.exp(-self.gamma * light_intensity)
    
    def move_firefly(self, firefly, target_firefly):
        attractiveness = self.attractiveness(target_firefly.light_intensity)
        firefly += attractiveness * (target_firefly.position - firefly) + self.alpha * np.random.uniform(-1, 1, self.dim)
        return np.clip(firefly, self.lower_bound, self.upper_bound)
    
    def __call__(self, func):
        for _ in range(self.budget):
            light_intensity = np.array([func(firefly) for firefly in self.population])
            sorted_indices = np.argsort(light_intensity)
            for i, idx in enumerate(sorted_indices):
                for j in range(i+1, self.population_size):
                    self.population[idx] = self.move_firefly(self.population[idx], self.population[sorted_indices[j]])
        return self.population[sorted_indices[0]]