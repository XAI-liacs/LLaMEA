import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9, strategy='best'):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
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
            a, b, c = np.random.choice(indices, 3, replace=False)

            if self.strategy == 'best':
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
            elif self.strategy == 'current-to-best':
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                best_idx = np.argmin([func(ind) for ind in population])
                if i % 3 == 0:
                    mutant = population[i] + F * (population[best_idx] - population[i]) + F * (population[a] - population[b])
                else:
                    mutant = population[a] + F * (population[b] - population[c])
            else:
                F = self.F
                CR = self.CR

            mutant = population[a] + F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt