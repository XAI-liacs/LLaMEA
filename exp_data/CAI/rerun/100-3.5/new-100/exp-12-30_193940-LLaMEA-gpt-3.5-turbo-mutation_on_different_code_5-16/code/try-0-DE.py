import numpy as np

class DE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                np.random.shuffle(idxs)
                a, b, c = pop[np.random.choice(idxs[:3], 3, replace=False)]

                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[j])

                f = func(trial)
                if f < fitness[j]:
                    pop[j] = trial
                    fitness[j] = f

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = pop[best_idx]

        return self.f_opt, self.x_opt