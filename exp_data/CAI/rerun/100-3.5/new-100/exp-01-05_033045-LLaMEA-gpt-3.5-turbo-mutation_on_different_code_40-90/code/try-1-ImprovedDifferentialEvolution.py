import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, F_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
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
            self.F *= self.F_decay
            self.F = max(self.F, self.F_min)
            self.F = min(self.F, self.F_max)
        return self.f_opt, self.x_opt