import numpy as np

class DynamicLearningOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full(dim, 0.5)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            
            for i in range(self.dim):
                mutation_rate = np.clip(self.mutation_rates[i] + np.random.normal(0, 0.1), 0.1, 0.9)
                ensemble_mutation_rate = np.mean(self.mutation_rates) + mutation_rate
                population[:, i] = best_individual[i] + ensemble_mutation_rate * np.random.standard_normal(self.budget)
            
            fitness = np.array([func(individual) for individual in population])
        
        return best_individual