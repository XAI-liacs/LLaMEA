import numpy as np

class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.5, CR_max=1.0, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
            CR = self.CR_min + (self.CR_max - self.CR_min) * np.random.rand()

            mutant = population[a] + F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

            self.F = max(self.F_min, min(self.F_max, self.F + self.adapt_rate * (F - self.F)))
            self.CR = max(self.CR_min, min(self.CR_max, self.CR + self.adapt_rate * (CR - self.CR)))

        return self.f_opt, self.x_opt