import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.1, CR_min=0.1, F_decay=0.9, CR_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Initial Differential weight
        self.CR = CR  # Initial Crossover probability
        self.F_min = F_min  # Minimum value for F
        self.CR_min = CR_min  # Minimum value for CR
        self.F_decay = F_decay  # Decay rate for F
        self.CR_decay = CR_decay  # Decay rate for CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            F_val = max(np.random.normal(self.F, 0.1), self.F_min)
            CR_val = max(np.random.normal(self.CR, 0.1), self.CR_min)

            mutant = population[a] + F_val * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_val
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            self.F = max(self.F * self.F_decay, self.F_min)
            self.CR = max(self.CR * self.CR_decay, self.CR_min)

        return self.f_opt, self.x_opt