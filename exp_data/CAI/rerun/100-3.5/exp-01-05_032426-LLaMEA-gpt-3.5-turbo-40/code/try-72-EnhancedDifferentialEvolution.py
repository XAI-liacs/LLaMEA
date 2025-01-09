import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lower=0.2, F_upper=0.8, CR_lower=0.2, CR_upper=1.0, strategy_adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.F_lower = F_lower  # Lower bound for F
        self.F_upper = F_upper  # Upper bound for F
        self.CR_lower = CR_lower  # Lower bound for CR
        self.CR_upper = CR_upper  # Upper bound for CR
        self.strategy_adaptation_rate = strategy_adaptation_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_i = np.clip(self.F + np.random.uniform(-self.strategy_adaptation_rate, self.strategy_adaptation_rate), self.F_lower, self.F_upper)
            CR_i = np.clip(self.CR + np.random.uniform(-self.strategy_adaptation_rate, self.strategy_adaptation_rate), self.CR_lower, self.CR_upper)

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