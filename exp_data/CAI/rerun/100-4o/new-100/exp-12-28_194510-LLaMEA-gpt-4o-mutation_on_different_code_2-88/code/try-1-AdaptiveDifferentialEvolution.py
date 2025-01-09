import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for i in range(self.budget - self.population_size):
            if i % (self.population_size) == 0:
                best_idx = np.argmin(fitness)
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

            idxs = np.arange(self.population_size)
            for j in range(self.population_size):
                a, b, c = population[np.random.choice(idxs[idxs != j], 3, replace=False)]
                self.mutation_factor = 0.5 + 0.3 * np.random.rand()  # Change: Dynamic adaptation of mutation factor
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover_mask, mutant, population[j])
                f_trial = func(trial)

                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial

        return self.f_opt, self.x_opt