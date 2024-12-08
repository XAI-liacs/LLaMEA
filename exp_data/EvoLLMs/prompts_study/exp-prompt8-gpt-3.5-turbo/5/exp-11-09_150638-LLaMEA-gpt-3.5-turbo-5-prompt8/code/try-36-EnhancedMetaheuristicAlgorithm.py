import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            fittest = population[sorted_indices[0]]
            pop_mean = np.mean(population, axis=0)
            diversity = np.mean(np.linalg.norm(population - pop_mean, axis=1))

            for i in range(self.budget):
                individual_mutation_strength = 5.0 / (1.0 + np.abs(fitness[i] - min(fitness)))

                mutated = population[i] + individual_mutation_strength * np.random.randn(self.dim)
                mutated_fitness = func(mutated)
                
                if mutated_fitness < fitness[i]:
                    population[i] = mutated
                    fitness[i] = mutated_fitness

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        return best_solution