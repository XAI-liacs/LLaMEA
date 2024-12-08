import numpy as np

class SelfAdaptiveMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rates = np.full(budget, 0.05)  # Initialize mutation rates
        
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutations = np.random.randn(self.budget, self.dim) * self.mutation_rates[:, np.newaxis]
            new_solutions = self.population + mutations
            new_fitness = [func(ind) for ind in new_solutions]
            improved_mask = new_fitness < fitness
            self.population[improved_mask] = new_solutions[improved_mask]
            self.mutation_rates[improved_mask] *= 1.2  # Increase mutation rate for improved solutions
            self.mutation_rates[~improved_mask] *= 0.8  # Decrease mutation rate for non-improved solutions
        return self.population[np.argmin([func(ind) for ind in self.population])]