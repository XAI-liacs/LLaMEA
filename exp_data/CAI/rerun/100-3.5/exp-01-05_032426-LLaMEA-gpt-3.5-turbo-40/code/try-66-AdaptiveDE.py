import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, F_lower=0.1, F_upper=0.9, CR_lower=0.1, CR_upper=0.9):
        self.budget = budget
        self.dim = dim
        self.F_lower = F_lower  # Lower bound for differential weight
        self.F_upper = F_upper  # Upper bound for differential weight
        self.CR_lower = CR_lower  # Lower bound for crossover probability
        self.CR_upper = CR_upper  # Upper bound for crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F_list = np.random.uniform(self.F_lower, self.F_upper, size=self.budget)
        CR_list = np.random.uniform(self.CR_lower, self.CR_upper, size=self.budget)
        
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            mutant = population[a] + F_list[i] * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_list[i]
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt