import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        
        for _ in range(self.budget):
            theta = np.random.uniform(0, np.pi)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.array([individual.dot(rotation_matrix) for individual in population])
            best_solution = population[np.argmin([func(individual) for individual in population])]
        
        return best_solution