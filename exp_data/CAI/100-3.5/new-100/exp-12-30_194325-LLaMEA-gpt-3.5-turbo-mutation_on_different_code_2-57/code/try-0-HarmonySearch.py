import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        for i in range(self.budget):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(0, self.budget)
                new_x = harmony_memory[idx]
                for j in range(self.dim):
                    if np.random.rand() < self.par:
                        new_x[j] = new_x[j] + np.random.uniform(-self.bw, self.bw)
                        new_x[j] = np.clip(new_x[j], func.bounds.lb, func.bounds.ub)
            else:
                new_x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(new_x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_x
                harmony_memory[i] = new_x

        return self.f_opt, self.x_opt