import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9, strategy='rand-to-best/1'):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.strategy = strategy
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]

            if self.strategy == 'rand-to-best/1':
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
            else:  # Default to 'rand/1' strategy
                a, b, c, d = np.random.choice(indices, 4, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c]) + self.F * (population[d] - population[i])

            crossover_mask = np.random.rand(self.dim) < self.CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            self.F = max(self.F_min, min(self.F_max, self.F + 0.01 * (f_offspring - self.f_opt)))
            self.CR = max(self.CR_min, min(self.CR_max, self.CR + 0.01 * (f_offspring - self.f_opt)))

        return self.f_opt, self.x_opt