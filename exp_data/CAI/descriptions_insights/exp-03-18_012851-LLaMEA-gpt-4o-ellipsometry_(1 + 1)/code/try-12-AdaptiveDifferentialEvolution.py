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
                # Randomly select three individuals different from i
                indices = np.random.choice([ind for ind in range(self.population_size) if ind != i], 3, replace=False)
                x1, x2, x3 = population[indices]

                # Mutation and crossover
                mutant = np.clip(x1 + self.f * (x2 - x3), lower_bound, upper_bound)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                # Break if budget is exhausted
                if budget_used >= self.budget:
                    break

            # Update population
            population = new_population

            # Dynamically adjust control parameters
            diversity = np.mean(np.std(population, axis=0))  # Measure population diversity
            self.f = np.random.uniform(0.4, 0.9) if diversity > 0.1 else np.random.uniform(0.5, 0.7)
            self.cr = np.random.uniform(0.85, 1.0)  # Adjusted crossover range

            # Dynamically adjust population size
            self.population_size = max(4, int(10 * self.dim * (1 - budget_used / self.budget)))

            # Elitism: Retain the best solution found so far
            best_indices = np.argsort(fitness)[:2]
            population[:2] = population[best_indices]
            fitness[:2] = fitness[best_indices]

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]