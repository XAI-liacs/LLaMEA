import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lb=0.4, F_ub=0.9, CR_lb=0.5, CR_ub=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_lb = F_lb  # Lower bound of F
        self.F_ub = F_ub  # Upper bound of F
        self.CR_lb = CR_lb  # Lower bound of CR
        self.CR_ub = CR_ub  # Upper bound of CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_val = np.random.uniform(self.F_lb, self.F_ub)
            CR_val = np.random.uniform(self.CR_lb, self.CR_ub)

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