import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.75, par=0.5, bw=0.1):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        for i in range(self.budget):
            new_solution = np.copy(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_solution[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    else:
                        rand_idx = np.random.choice(self.dim)
                        new_solution[j] = harmony_memory[rand_idx] + np.random.uniform(-self.bw, self.bw)
            f = func(new_solution)
            if f < func(harmony_memory):
                harmony_memory = np.copy(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution

        return self.f_opt, self.x_opt