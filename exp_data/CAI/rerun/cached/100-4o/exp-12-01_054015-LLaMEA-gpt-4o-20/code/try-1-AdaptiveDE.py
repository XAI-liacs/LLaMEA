import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(range(self.pop_size), 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                # Ensure trial vector is within bounds
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                f_trial = func(trial)
                evaluations += 1

                # Update population with competitive fitness-based selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]