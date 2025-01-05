import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_lb=0.2, F_ub=0.8, CR_lb=0.1, CR_ub=0.9, strategy='rand-to-best/1'):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub
        self.strategy = strategy
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F = np.full(self.budget, self.F_init)
        CR = np.full(self.budget, self.CR_init)

        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            if self.strategy == 'rand-to-best/1':
                a, b, c = np.random.choice(indices, 3, replace=False)
                x_best = population[np.argmin([func(ind) for ind in population])]
                mutant = population[a] + F[i] * (x_best - population[a]) + F[i] * (population[b] - population[c])
            else:
                # Add more strategies here for exploration

            crossover_mask = np.random.rand(self.dim) < CR[i]
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
                if f_offspring < func(population[i]):
                    population[i] = offspring
                    F[i] = np.clip(F[i] + 0.01, self.F_lb, self.F_ub)
                    CR[i] = np.clip(CR[i] + 0.05, self.CR_lb, self.CR_ub)

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt