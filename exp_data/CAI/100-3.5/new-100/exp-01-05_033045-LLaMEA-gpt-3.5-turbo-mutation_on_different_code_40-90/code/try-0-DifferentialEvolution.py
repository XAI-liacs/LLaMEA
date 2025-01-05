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
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            idxs = np.random.choice(self.budget, 3, replace=False)
            a, b, c = population[idxs]
            mutant = a + self.F * (b - c)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            f_trial = func(trial)
            if f_trial < func(population[i]):
                population[i] = trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
        return self.f_opt, self.x_opt