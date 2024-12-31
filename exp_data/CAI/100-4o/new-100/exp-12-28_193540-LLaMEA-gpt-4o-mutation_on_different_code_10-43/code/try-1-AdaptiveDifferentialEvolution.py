import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evals = self.population_size

        while evals < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_1, x_2, x_3 = population[indices]
                mutant_vector = np.clip(x_1 + self.mutation_factor * (x_2 - x_3), self.bounds[0], self.bounds[1])
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                evals += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

                # Update best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial_vector

                if evals >= self.budget:
                    break

            population = new_population

            # Adaptive parameter tuning
            self.mutation_factor = 0.5 + 0.2 * (1 - evals / self.budget)
            self.crossover_rate = 0.7 + 0.1 * (self.f_opt / (self.f_opt + np.mean(fitness)))

            # Dynamic population resizing
            if evals / self.budget > 0.5:
                self.population_size = max(4, int(self.population_size * 0.9))
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

        return self.f_opt, self.x_opt