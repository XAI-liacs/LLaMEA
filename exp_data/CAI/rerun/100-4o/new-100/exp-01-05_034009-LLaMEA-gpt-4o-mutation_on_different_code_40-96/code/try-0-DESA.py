import numpy as np

class DESA:
    def __init__(self, budget=10000, dim=10, pop_size=50, f=0.8, cr=0.9, t0=1.0, cooling_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.t0 = t0
        self.cooling_rate = cooling_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        for evals in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    temp = self.t0 * (self.cooling_rate ** (evals / self.pop_size))
                    if np.random.rand() < np.exp((fitness[i] - f_trial) / temp):
                        population[i] = trial
                        fitness[i] = f_trial

        return self.f_opt, self.x_opt