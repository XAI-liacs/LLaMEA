import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness_values)
            elite_index = sorted_indices[0]
            elite = self.population[elite_index]
            
            new_population = []
            for i in range(self.budget):
                idx1, idx2 = np.random.choice(self.budget, 2, replace=False)
                parent1 = self.population[idx1]
                parent2 = self.population[idx2]
                beta = np.random.uniform(0, 1, self.dim)
                mask = np.random.choice([0, 1], self.dim)
                child = np.where(mask, parent1, parent2) + beta * (parent1 - parent2)
                new_population.append(child)
            
            new_population[elite_index] = elite
            self.population = np.array(new_population)
        
        best_index = np.argmin([func(individual) for individual in self.population])
        best_solution = self.population[best_index]
        return best_solution