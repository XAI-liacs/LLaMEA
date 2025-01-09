import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F_init
        self.CR = CR_init
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_min = 0.1
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.F_decay = 0.9
        self.CR_decay = 0.8

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            mutant = population[a] + self.F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < self.CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
                
                if np.random.rand() < 0.5:  # Dynamic adaptation of F and CR based on success
                    self.F = max(self.F_min, min(self.F_max, self.F * self.F_decay))
                else:
                    self.CR = max(self.CR_min, min(self.CR_max, self.CR * self.CR_decay))

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt