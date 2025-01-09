import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_min=0.2, CR_max=1.0, F_decay=0.9, CR_growth=1.1):
        self.budget = budget
        self.dim = dim
        self.F = F_init  # Differential weight
        self.CR = CR_init  # Crossover probability
        self.F_min = F_min
        self.CR_max = CR_max
        self.F_decay = F_decay
        self.CR_growth = CR_growth
        self.f_opt = np.Inf
        self.x_opt = None
        self.generation = 0

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

            # Adaptive control of mutation and crossover parameters
            if self.generation % 10 == 0:
                self.F = max(self.F * self.F_decay, self.F_min)
                self.CR = min(self.CR * self.CR_growth, self.CR_max)
            
            self.generation += 1

        return self.f_opt, self.x_opt