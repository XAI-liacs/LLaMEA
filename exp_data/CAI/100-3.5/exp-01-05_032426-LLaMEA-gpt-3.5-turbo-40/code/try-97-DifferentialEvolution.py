import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lb=0.1, F_ub=0.9, F_decay=0.9, CR_lb=0.1, CR_ub=0.9, CR_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_lb = F_lb  # Lower bound for F
        self.F_ub = F_ub  # Upper bound for F
        self.F_decay = F_decay  # Decay factor for F
        self.CR_lb = CR_lb  # Lower bound for CR
        self.CR_ub = CR_ub  # Upper bound for CR
        self.CR_decay = CR_decay  # Decay factor for CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_current = np.clip(self.F, self.F_lb, self.F_ub)
            CR_current = np.clip(self.CR, self.CR_lb, self.CR_ub)

            mutant = population[a] + F_current * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_current
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring
            
            self.F *= self.F_decay
            self.CR *= self.CR_decay

        return self.f_opt, self.x_opt