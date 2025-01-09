import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        F_lower, F_upper = 0.1, 0.9
        CR_lower, CR_upper = 0.1, 0.9

        for _ in range(self.budget // self.pop_size):
            F = np.random.uniform(F_lower, F_upper)
            CR = np.random.uniform(CR_lower, CR_upper)

            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = population[a] + F * (population[b] - population[c])
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                f_trial = func(trial)

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                population[i] = trial if f_trial < func(population[i]) else population[i]

        return self.f_opt, self.x_opt