import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, adapt_rate=0.05):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.adapt_rate = adapt_rate  # Adaptive control parameter
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_i = np.clip(self.F + np.random.normal(0, self.adapt_rate), 0, 2)  # Adaptive F for individual
            CR_i = np.clip(self.CR + np.random.normal(0, self.adapt_rate), 0, 1)  # Adaptive CR for individual

            mutant = population[a] + F_i * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_i
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt