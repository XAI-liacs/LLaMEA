import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_decay=0.9, CR_growth=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F_init  # Differential weight
        self.CR = CR_init  # Crossover probability
        self.F_decay = F_decay  # Decay factor for F
        self.CR_growth = CR_growth  # Growth factor for CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_i = np.clip(self.F + np.random.normal(0, 0.1), 0, 2)  # Adaptive control for F
            CR_i = np.clip(self.CR + np.random.normal(0, 0.1), 0, 1)  # Adaptive control for CR

            mutant = population[a] + F_i * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_i
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            self.F *= self.F_decay
            self.CR += self.CR_growth

        return self.f_opt, self.x_opt