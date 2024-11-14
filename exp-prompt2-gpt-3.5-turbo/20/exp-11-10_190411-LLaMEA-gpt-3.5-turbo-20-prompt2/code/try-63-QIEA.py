import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def adaptive_quantum_rotation(self, population, func):
        best_solution = population[np.argmin(func(population))]
        for _ in range(self.budget):
            # Adaptive quantum rotation gates based on fitness improvement
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.dot(population, rotation_matrix)
            
            # Apply evolutionary operators
            offspring = population + np.random.normal(0, 0.1, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        return best_solution

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        
        best_solution = self.adaptive_quantum_rotation(population, func)
        
        return best_solution