import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim  # Dynamic initial population size based on dimension
        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_probability = 0.7  # Initial crossover probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.initial_population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_population_size
        
        while evaluations < self.budget:
            current_population_size = int(self.initial_population_size * (1 - evaluations / self.budget)) + 1  # Progressive resizing
            for i in range(current_population_size):
                indices = np.arange(current_population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            self.mutation_factor = 0.8 * (1 - evaluations / self.budget) + 0.1
            self.crossover_probability = 0.9 * (evaluations / self.budget) + 0.1

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]