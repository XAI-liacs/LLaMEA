import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_min, self.F_max = 0.2, 0.8
        self.CR_min, self.CR_max = 0.5, 1.0

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_val = np.random.uniform(self.F_min, self.F_max)
            CR_val = np.random.uniform(self.CR_min, self.CR_max)

            mutant = population[a] + F_val * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_val
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt