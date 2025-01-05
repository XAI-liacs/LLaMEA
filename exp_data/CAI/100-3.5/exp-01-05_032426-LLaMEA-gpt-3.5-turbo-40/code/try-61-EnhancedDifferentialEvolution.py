import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_lb=0.1, F_ub=0.9, CR_lb=0.1, CR_ub=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F_init
        self.CR = CR_init
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_current = max(self.F_lb, min(self.F_ub, self.F + np.random.uniform(-self.adapt_rate, self.adapt_rate)))
            CR_current = max(self.CR_lb, min(self.CR_ub, self.CR + np.random.uniform(-self.adapt_rate, self.adapt_rate)))

            mutant = population[a] + F_current * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_current
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt