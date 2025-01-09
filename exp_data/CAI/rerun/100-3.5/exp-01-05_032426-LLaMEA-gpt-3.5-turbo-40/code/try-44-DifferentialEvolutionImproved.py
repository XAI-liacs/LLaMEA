import numpy as np

class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F_init=0.5, F_l=0.2, F_u=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.F_l = F_l  # Lower bound for F
        self.F_u = F_u  # Upper bound for F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F = np.random.uniform(self.F_l, self.F_u)  # Adaptive scaling factor
            mutant = population[a] + F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < self.CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt