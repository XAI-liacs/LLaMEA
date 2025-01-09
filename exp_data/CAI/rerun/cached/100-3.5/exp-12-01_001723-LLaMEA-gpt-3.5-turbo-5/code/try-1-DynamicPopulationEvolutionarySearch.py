import numpy as np

class DynamicPopulationEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.max_pop_size = 100
        self.mutation_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        evaluations = 0
        
        while evaluations < self.budget:
            fitness = np.array([func(individual) for individual in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            
            new_population = [best_individual]
            for _ in range(self.pop_size - 1):
                parent_idx = np.random.randint(self.pop_size)
                parent = population[parent_idx]
                mutation = np.random.uniform(-1.0, 1.0, size=self.dim) * self.mutation_rate
                child = np.clip(parent + mutation, -5.0, 5.0)
                new_population.append(child)
            
            population = np.array(new_population)
            evaluations += self.pop_size
            
            if self.pop_size < self.max_pop_size:
                self.pop_size += 10
            
        return best_individual