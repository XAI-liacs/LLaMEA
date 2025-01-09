import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.F_min, self.F_max = 0.5, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Adaptive parameter setting
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                # Mutation and crossover
                parents = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[parents]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                # Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < CR
                trial[crossover_mask] = mutant[crossover_mask]
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]