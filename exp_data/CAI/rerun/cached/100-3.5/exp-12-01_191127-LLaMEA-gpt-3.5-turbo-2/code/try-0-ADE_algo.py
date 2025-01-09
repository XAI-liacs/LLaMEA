import numpy as np

class ADE_algo:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.np = 10
        self.CR = 0.9
        self.F_min = 0.1
        self.F_max = 0.9
        self.pop = np.random.uniform(-5.0, 5.0, (self.np, self.dim))

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            for i in range(self.np):
                idxs = [idx for idx in range(self.np) if idx != i]
                a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.normal(self.CR, 0.1)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, self.pop[i])
                if func(trial) < func(self.pop[i]):
                    self.pop[i] = trial
                evals += 1
                if evals >= self.budget:
                    break
        best_idx = np.argmin([func(ind) for ind in self.pop])
        return self.pop[best_idx]