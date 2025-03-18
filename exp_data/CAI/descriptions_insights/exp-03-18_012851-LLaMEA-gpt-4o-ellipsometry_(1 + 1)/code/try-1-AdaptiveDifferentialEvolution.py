import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.5  # Initial differential weight
        self.cr = 0.9  # Initial crossover probability

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        # Main optimization loop
        while budget_used < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = np.random.choice([ind for ind in range(self.population_size) if ind != i], 3, replace=False)
                x1, x2, x3 = population[indices]

                # Mutation and crossover
                mutant = np.clip(x1 + self.f * (x2 - x3), lower_bound, upper_bound)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])

                # Opposition-based learning
                opposite = lower_bound + upper_bound - trial
                trial_fitness = func(trial)
                opposite_fitness = func(opposite)
                budget_used += 2

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                elif opposite_fitness < fitness[i]:
                    new_population[i] = opposite
                    fitness[i] = opposite_fitness

                if budget_used >= self.budget:
                    break

            # Update population
            population = new_population

            # Dynamically adjust control parameters
            self.f = np.random.uniform(0.4, 0.9)
            self.cr = np.random.uniform(0.8, 1.0)

        best_index = np.argmin(fitness)
        return population[best_index]