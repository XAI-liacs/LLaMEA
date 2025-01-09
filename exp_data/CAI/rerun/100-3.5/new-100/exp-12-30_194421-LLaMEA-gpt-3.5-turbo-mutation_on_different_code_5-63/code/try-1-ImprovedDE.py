import numpy as np

class ImprovedDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, history_len=10):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.history_len = history_len
        self.history = [F] * history_len
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                np.random.shuffle(idxs)
                a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]

                self.F = np.mean(self.history)
                mutant = np.clip(a + self.F * (b - c) + self.F * (2 * np.random.rand(self.dim) - 1), bounds[0], bounds[1])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[j])

                f = func(trial)
                if f < func(population[j]):
                    population[j] = trial
                    self.history.append(self.F)
                    self.history.pop(0)

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

        return self.f_opt, self.x_opt