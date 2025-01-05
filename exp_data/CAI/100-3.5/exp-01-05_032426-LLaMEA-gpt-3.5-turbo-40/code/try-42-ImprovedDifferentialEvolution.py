import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, F_decay=0.99, CR_init=0.9, CR_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F_init
        self.F_init = F_init
        self.F_decay = F_decay
        self.CR = CR_init
        self.CR_init = CR_init
        self.CR_decay = CR_decay
        self.f_opt = np.Inf
        self.x_opt = None

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
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            self.F *= self.F_decay
            self.CR *= self.CR_decay

        return self.f_opt, self.x_opt