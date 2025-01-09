import numpy as np

class DynamicDE:
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

        for i in range(self.budget):
            F = max(0.1, self.F + 0.8 * np.random.rand())  # Dynamic F
            CR = max(0.1, self.CR + 0.8 * np.random.rand())  # Dynamic CR
            
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])

                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[j])

                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt