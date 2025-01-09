import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_min=0.2, F_max=0.8, CR_min=0.2, CR_max=0.9):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])

        def clip(x):
            return np.clip(x, bounds[0], bounds[1])

        pop = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))

        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                mutant = clip(a + F * (b - c))
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                f_trial = func(trial)
                if f_trial < func(pop[i]):
                    pop[i] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt