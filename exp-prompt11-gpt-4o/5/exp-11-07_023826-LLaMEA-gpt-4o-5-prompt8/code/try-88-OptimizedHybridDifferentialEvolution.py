import numpy as np

class OptimizedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5  # scaling factor
        self.cr = 0.9  # crossover probability
        self.strategy_switch_ratio = 0.5  # ratio to switch between exploration and exploitation

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            indices = np.arange(self.pop_size)
            np.random.shuffle(indices)
            for idx in range(0, self.pop_size, 3):
                if num_evaluations >= self.budget:
                    break

                i = indices[idx]
                # Mutation - choose three unique donors
                a, b, c = indices[idx:idx+3]
                if len({a, b, c, i}) < 4:
                    continue  # ensure unique selection

                mutant_vector = population[a] + self.f * (population[b] - population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

            # Adaptive mutation strategy based on evaluations
            if num_evaluations / self.budget > self.strategy_switch_ratio:
                self.f = np.random.uniform(0.4, 0.9)  # switch to more exploitation

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]