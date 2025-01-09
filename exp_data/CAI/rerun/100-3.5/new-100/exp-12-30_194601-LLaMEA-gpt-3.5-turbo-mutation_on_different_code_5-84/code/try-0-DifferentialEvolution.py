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
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))

        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                target = population[i]
                indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = population[indices]

                mutant = a + self.F * (b - c)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, target)

                f_target = func(target)
                f_trial = func(trial)

                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt