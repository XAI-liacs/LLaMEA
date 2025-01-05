import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.f = f
        self.cr = cr

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (-5.0, 5.0)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                np.random.shuffle(idxs)
                r1, r2, r3 = population[np.random.choice(idxs[:3], 3, replace=False)]
                mutant = np.clip(population[r1] + self.f * (population[r2] - population[r3]), bounds[0], bounds[1])

                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[j])

                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.f_opt:
                self.f_opt = fitness[min_idx]
                self.x_opt = population[min_idx]

        return self.f_opt, self.x_opt