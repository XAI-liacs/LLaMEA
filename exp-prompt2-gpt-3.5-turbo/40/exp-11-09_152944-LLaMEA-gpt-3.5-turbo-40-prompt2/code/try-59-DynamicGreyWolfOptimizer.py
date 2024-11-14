import numpy as np

class DynamicGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3
                
                population_fitness = [func(ind) for ind in population]
                sorted_indices = np.argsort(population_fitness)
                alpha, beta, delta = population[sorted_indices[:3]]
                
                # Dynamic alpha update based on fitness evaluation
                if population_fitness[sorted_indices[0]] < population_fitness[sorted_indices[1]]:
                    a = 2 - 2 * (_ / self.budget) * (population_fitness[sorted_indices[0]] / population_fitness[sorted_indices[1]])
                    alpha = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - population[sorted_indices[0]])
                
            return alpha

        return optimize()