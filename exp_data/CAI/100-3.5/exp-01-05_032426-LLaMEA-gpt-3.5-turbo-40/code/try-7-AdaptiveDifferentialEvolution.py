import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_init=0.5, CR_init=0.9, F_lb=0.1, F_ub=0.9, CR_lb=0.1, CR_ub=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_init = F_init  # Initial F
        self.CR_init = CR_init  # Initial CR
        self.F_lb = F_lb  # F lower bound
        self.F_ub = F_ub  # F upper bound
        self.CR_lb = CR_lb  # CR lower bound
        self.CR_ub = CR_ub  # CR upper bound
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_current = self.F_init + (self.F - self.F_init) * i / self.budget
            CR_current = self.CR_init + (self.CR - self.CR_init) * i / self.budget

            mutant = population[a] + F_current * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_current
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt