import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Dynamic population size based on dimension
        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_probability = 0.7  # Initial crossover probability

    def __call__(self, func):
        # Initialize population within bounds
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = indices[indices != i]  # Exclude the current index
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Check if budget exceeded
                if evaluations >= self.budget:
                    break

            # Dynamic adaptation of differential evolution parameters
            self.mutation_factor = 0.8 * (1 - np.std(fitness) / np.mean(fitness)) + 0.1
            self.crossover_probability = 0.9 * (1 - np.std(fitness) / np.mean(fitness)) + 0.1  # Line changed

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]