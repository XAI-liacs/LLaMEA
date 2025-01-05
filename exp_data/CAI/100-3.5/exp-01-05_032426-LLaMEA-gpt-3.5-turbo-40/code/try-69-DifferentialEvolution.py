import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_min=0.4, F_max=0.9, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min  
        self.F_max = F_max  
        self.CR = CR  
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
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