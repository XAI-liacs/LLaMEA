import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])

        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                idxs = idxs[idxs != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])

                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt