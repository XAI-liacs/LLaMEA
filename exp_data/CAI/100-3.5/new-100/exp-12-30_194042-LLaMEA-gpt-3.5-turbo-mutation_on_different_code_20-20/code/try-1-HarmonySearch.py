import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, par_min=0.01, par_max=0.5, par_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.par_min = par_min
        self.par_max = par_max
        self.par_decay = par_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        for _ in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = harmony_memory[j]
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.normal(0, self.bw)

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony

            self.par = max(self.par * self.par_decay, self.par_min) if f < self.f_opt else min(self.par * (1/self.par_decay), self.par_max)

        return self.f_opt, self.x_opt