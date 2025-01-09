import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_decay=0.9, CR_inc=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F  
        self.CR = CR  
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_decay = F_decay
        self.CR_inc = CR_inc

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_current = np.random.normal(self.F, 0.1)
            CR_current = np.clip(np.random.normal(self.CR, 0.1), 0, 1)

            mutant = population[a] + F_current * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_current
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            self.F *= self.F_decay
            self.CR += self.CR_inc

        return self.f_opt, self.x_opt