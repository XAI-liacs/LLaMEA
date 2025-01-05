import numpy as np

class ImprovedDE:
    def __init__(self, budget=10000, dim=10, F_init=0.5, F_lower=0.1, F_upper=0.9, CR_init=0.9, CR_lower=0.5, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_init = CR_init
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F = self.F_init
        CR = self.CR_init
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            mutant = population[a] + F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            F = max(self.F_lower, min(self.F_upper, F + 0.01 * np.random.randn()))
            CR = max(self.CR_lower, min(self.CR_upper, CR + 0.01 * np.random.randn()))

        return self.f_opt, self.x_opt