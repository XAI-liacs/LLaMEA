import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def quantum_rotation(self, x, alpha):
        return x * np.exp(1j * alpha)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = np.zeros(self.dim)
        
        for _ in range(self.budget):
            rotated_population = [self.quantum_rotation(individual, np.pi / 2) for individual in population]
            fitness_values = [func(individual) for individual in rotated_population]
            best_idx = np.argmin(fitness_values)
            best_solution = rotated_population[best_idx]
            
            # Instead of reinitializing the population, update the existing population itself
            for i in range(self.budget):
                if i != best_idx:  # Avoid unnecessary calculations for the best individual
                    population[i] = np.random.uniform(-5.0, 5.0, self.dim)
        
        return best_solution