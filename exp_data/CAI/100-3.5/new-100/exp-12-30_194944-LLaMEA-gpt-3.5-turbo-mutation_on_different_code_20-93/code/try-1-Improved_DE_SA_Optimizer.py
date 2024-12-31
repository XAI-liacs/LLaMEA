import numpy as np

class Improved_DE_SA_Optimizer:
    def __init__(self, budget=10000, dim=10, de_cr=0.5, de_f=0.8, sa_temp=1.0, sa_cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.de_cr = de_cr
        self.de_f = de_f
        self.sa_temp = sa_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.f_opt = np.Inf
        self.x_opt = None
        self.cr_adapt = 0.1
        self.f_adapt = 0.1

    def __call__(self, func):
        def clip(x, l, u):
            return np.maximum(np.minimum(x, u), l)

        x = np.random.uniform(func.bounds.lb, func.bounds.ub)
        f = func(x)
        if f < self.f_opt:
            self.f_opt = f
            self.x_opt = x

        for i in range(self.budget - 1):
            self.de_cr = max(0.1, min(0.9, self.de_cr + np.random.normal(0, self.cr_adapt)))
            self.de_f = max(0.1, min(0.9, self.de_f + np.random.normal(0, self.f_adapt)))

            mutant = clip(self.x_opt + self.de_f * (x - self.x_opt), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.de_cr
            trial_x = np.where(cross_points, mutant, x)
            trial_f = func(trial_x)

            if trial_f < f or np.random.rand() < np.exp((f - trial_f) / self.sa_temp):
                x = trial_x
                f = trial_f

            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            self.sa_temp *= self.sa_cooling_rate

        return self.f_opt, self.x_opt