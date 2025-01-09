import numpy as np

class AdaptiveCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.step_size = 0.5
        self.nests = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            new_nests = np.clip(self.nests + np.random.normal(0, self.step_size, size=(self.population_size, self.dim)), -5.0, 5.0)
            new_nests_fitness = np.array([func(nest) for nest in new_nests])
            sorted_indices = np.argsort(new_nests_fitness)
            self.nests[sorted_indices[:-1]] += np.random.normal(0, 0.1, size=(self.population_size-1, self.dim))
            self.step_size *= 0.99
        return self.nests[sorted_indices[0]]