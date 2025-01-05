import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lower=0.1, F_upper=0.9, CR_lower=0.1, CR_upper=1.0, strategy='rand-to-best/1/bin'):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.strategy = strategy
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]

            if self.strategy == 'rand/1/bin':
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
            elif self.strategy == 'rand-to-best/1/bin':
                a, b, c = np.random.choice(indices, 3, replace=False)
                best = np.argmin([func(p) for p in population])
                mutant = population[i] + self.F * (population[best] - population[i])
            else:
                raise ValueError("Invalid strategy provided")

            crossover_mask = np.random.rand(self.dim) < self.CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring
                
            # Adaptive control of F and CR
            self.F = max(self.F_lower, min(self.F_upper, self.F + np.random.normal(0, 0.1)))
            self.CR = max(self.CR_lower, min(self.CR_upper, self.CR + np.random.normal(0, 0.1))

        return self.f_opt, self.x_opt