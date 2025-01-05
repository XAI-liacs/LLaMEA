import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lower=0.4, F_upper=0.9, CR_lower=0.5, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_lower = F_lower  # Lower bound for F
        self.F_upper = F_upper  # Upper bound for F
        self.CR_lower = CR_lower  # Lower bound for CR
        self.CR_upper = CR_upper  # Upper bound for CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_val = np.random.uniform(self.F_lower, self.F_upper)
            mutant = population[a] + F_val * (population[b] - population[c])

            CR_val = np.random.uniform(self.CR_lower, self.CR_upper)
            crossover_mask = np.random.rand(self.dim) < CR_val
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt