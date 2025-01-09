import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim  # Population size
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # Crossover
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, mutant, pop[j])

                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if i < self.budget - 1:
                    pop[j] = trial

        return self.f_opt, self.x_opt