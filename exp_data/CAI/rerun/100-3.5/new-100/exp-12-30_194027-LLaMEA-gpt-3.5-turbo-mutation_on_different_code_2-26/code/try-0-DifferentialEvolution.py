import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, CR=0.9, F=0.5):
        self.budget = budget
        self.dim = dim
        self.CR = CR
        self.F = F
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def mutate(x_pop, F):
            n_pop = x_pop.shape[0]
            idxs = np.random.choice(n_pop, size=(n_pop, 3), replace=False)
            a, b, c = x_pop[idxs[:, 0]], x_pop[idxs[:, 1]], x_pop[idxs[:, 2]]
            return a + F * (b - c)

        def crossover(x_mutant, x_target, CR):
            mask = np.random.uniform(size=x_mutant.shape) < CR
            return np.where(mask, x_mutant, x_target)

        bounds = np.repeat([[func.bounds.lb], [func.bounds.ub]], self.dim, axis=0)
        x_pop = np.random.uniform(bounds[0], bounds[1], size=(self.budget, self.dim))

        for i in range(self.budget):
            x_mutant = mutate(x_pop, self.F)
            x_trial = crossover(x_mutant, x_pop[i], self.CR)

            f_trial = func(x_trial)
            if f_trial < func(x_pop[i]):
                x_pop[i] = x_trial

            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = x_trial

        return self.f_opt, self.x_opt