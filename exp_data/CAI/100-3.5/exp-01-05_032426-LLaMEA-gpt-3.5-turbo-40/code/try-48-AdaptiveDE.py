import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lb=0.1, F_ub=0.9, CR_lb=0.1, CR_ub=0.9, strategy_adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_lb = F_lb  # Lower bound for F
        self.F_ub = F_ub  # Upper bound for F
        self.CR_lb = CR_lb  # Lower bound for CR
        self.CR_ub = CR_ub  # Upper bound for CR
        self.strategy_adaptation_rate = strategy_adaptation_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def adapt_parameters(self, improvement):
        self.F = min(max(self.F * (1 + self.strategy_adaptation_rate * improvement), self.F_lb), self.F_ub)
        self.CR = min(max(self.CR * (1 + self.strategy_adaptation_rate * improvement), self.CR_lb), self.CR_ub)

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
                improvement = 1
            else:
                improvement = 0

            self.adapt_parameters(improvement)

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt