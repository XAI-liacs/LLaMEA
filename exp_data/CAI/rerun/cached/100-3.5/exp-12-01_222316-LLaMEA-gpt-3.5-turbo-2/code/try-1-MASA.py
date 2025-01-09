import numpy as np

class MASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = 0.1

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, size=self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            population = [best_solution + np.random.uniform(-1, 1, size=self.dim) * self.mutation_rate 
                          for _ in range(self.pop_size)]
            fitness_values = [func(individual) for individual in population]
            
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness_values[best_idx]

            self.pop_size = max(1, min(100, int(self.pop_size * 1.1)))
            self.mutation_rate = max(0.01, min(1.0, self.mutation_rate * 1.01))

        return best_solution