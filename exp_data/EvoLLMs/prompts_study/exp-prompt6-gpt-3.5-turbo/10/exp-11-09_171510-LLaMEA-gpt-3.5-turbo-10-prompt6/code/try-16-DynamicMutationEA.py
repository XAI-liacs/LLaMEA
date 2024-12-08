import numpy as np

class DynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_strengths = np.ones(dim) * 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]

            new_population = []
            for individual in self.population:
                mutant = individual + np.random.normal(0, self.mutation_strengths)
                new_population.append(mutant)

            self.population = np.array(new_population)
            self.mutation_strengths *= 0.95 + 0.05 * np.random.uniform(-1, 1, self.dim)  # Dynamically adapt mutation strengths

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual