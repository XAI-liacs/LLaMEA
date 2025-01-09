import numpy as np

class SwarmBiogeographyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.migration_probability = 0.2
        self.mutation_probability = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        eval_count = 0
        fitness = np.array([func(ind) for ind in self.population])
        eval_count += self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = self.population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            new_population = np.copy(self.population)

            for i in range(self.population_size):
                if np.random.rand() < self.migration_probability:
                    migrants = np.random.permutation(self.population_size)
                    donor = migrants[0] if migrants[0] != i else migrants[1]
                    new_population[i] = self.migrate(self.population[i], self.population[donor])

                if np.random.rand() < self.mutation_probability:
                    new_population[i] = self.mutate(new_population[i])
                
            new_fitness = np.array([func(ind) for ind in new_population])
            eval_count += self.population_size
            combined = np.vstack((self.population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))

            selected_indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = combined[selected_indices]
            fitness = combined_fitness[selected_indices]

            current_best_index = np.argmin(fitness)
            current_best_solution = self.population[current_best_index]
            current_best_fitness = fitness[current_best_index]

            if current_best_fitness < best_fitness:
                best_solution = current_best_solution
                best_fitness = current_best_fitness

        return best_solution

    def migrate(self, habitat, donor):
        return habitat + np.random.rand(self.dim) * (donor - habitat)

    def mutate(self, solution):
        mutation_strength = 0.1
        return solution + mutation_strength * np.random.normal(0, 1, self.dim)